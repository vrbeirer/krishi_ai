# Krishi.ai local knowledge sources

Add trusted agriculture documents as `.md`, `.txt`, `.rst`, or `.csv` files in this directory. Good sources include current ICAR publications, state agriculture university/KVK guidance, and official government advisories. Include the title, publisher, publication/update date, crop, region, and URL near the top of each file so responses can name the source.

Keep content current and crop/region-specific. Do not add unverified social-media advice, copied AI answers, or product marketing claims. The backend chunks these files and retrieves relevant excerpts for RAG. PDFs are not indexed in this starter version: convert verified PDFs to text/Markdown before adding them.

No authoritative crop-specific recommendations are bundled by default. This is intentional: generic fertilizer rates or pesticide doses can be unsafe when crop, soil, pest, region and growth stage are unknown.
