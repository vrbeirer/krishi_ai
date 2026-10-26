import unittest

import app as krishi_app


class KrishiApiTests(unittest.TestCase):
    def setUp(self):
        krishi_app.app.config["TESTING"] = True
        self.client = krishi_app.app.test_client()

    def test_home_serves_single_file_frontend(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Krishi.ai", response.data)
        self.assertIn(b"/translate", response.data)

    def test_chat_rejects_invalid_json(self):
        response = self.client.post("/chat", data="not-json", content_type="text/plain")
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()["success"])

    def test_chat_rejects_empty_message(self):
        response = self.client.post("/chat", json={"message": "   ", "language": "en"})
        self.assertEqual(response.status_code, 400)

    def test_translate_english_is_passthrough(self):
        response = self.client.post("/translate", json={"text": "Hello", "language": "en"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"success": True, "translated": "Hello"})

    def test_unknown_language_falls_back_to_english(self):
        self.assertEqual(krishi_app.normalize_language("unknown"), "en")

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_location_extraction_needs_explicit_place(self):
        self.assertEqual(krishi_app._extract_location("Weather in Chikodi, Karnataka today?"), "Chikodi, Karnataka")
        self.assertEqual(krishi_app._extract_location("Chikodi मध्ये आज पाऊस पडेल का?"), "Chikodi")
        self.assertEqual(krishi_app._extract_location("What is the weather forecast for farming today?"), "")

    def test_commodity_detection(self):
        self.assertEqual(krishi_app._extract_commodity("What are tomato prices in Pune?"), "tomato")
        self.assertEqual(krishi_app._extract_commodity("बटाटा भाव आज काय आहे?"), "potato")

    def test_history_is_bounded_and_role_filtered(self):
        history = [
            {"role": "user", "content": "question"},
            {"role": "system", "content": "do not accept"},
            {"role": "assistant", "content": "answer"},
        ]
        clean = krishi_app._safe_history(history)
        self.assertEqual([item["role"] for item in clean], ["user", "assistant"])

    def test_fallback_does_not_dump_raw_knowledge_chunks(self):
        raw_note = {"source": "agriculture_basics.md", "text": "# Generic notes\nSoil test general details"}
        answer = krishi_app._local_fallback("Which crops suit Kolhapur?", "en", [raw_note])
        self.assertIn("AI service", answer)
        self.assertNotIn("Soil test general details", answer)
        self.assertNotIn("agriculture_basics.md", answer)

    def test_frontend_has_explicit_audio_and_new_chat_controls(self):
        response = self.client.get("/")
        body = response.get_data(as_text=True)
        self.assertIn('id="newChatButton"', body)
        self.assertIn('krishi-speak', body)
        self.assertIn("renderMarkdown", body)
        self.assertNotIn("speak(data.response,language)", body)

    def test_frontend_has_previous_chat_and_voice_toggle(self):
        response = self.client.get("/")
        body = response.get_data(as_text=True)
        self.assertIn('id="historyButton"', body)
        self.assertIn('id="historyDialog"', body)
        self.assertIn("recognition.continuous=true", body)
        self.assertIn("recognition.interimResults=true", body)
        self.assertIn("krishi-voice-listening", body)
        self.assertIn("SAVED_CHATS_KEY", body)

    def test_broken_model_answer_detector(self):
        self.assertTrue(krishi_app._looks_like_broken_answer("User Safety: safe"))
        self.assertTrue(krishi_app._looks_like_broken_answer("**सेवन क"))
        self.assertFalse(krishi_app._looks_like_broken_answer("Check soil moisture before irrigating."))


if __name__ == "__main__":
    unittest.main()
