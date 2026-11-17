import os
import tempfile
import unittest
from novasearch.crawler import FileCrawler


class TestFileCrawler(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = self.temp_dir.name

        # Create sample files
        with open(os.path.join(self.root, "doc.md"), "w") as f:
            f.write("# Markdown Document\nTesting crawler.")

        with open(os.path.join(self.root, "script.py"), "w") as f:
            f.write("def main(): pass")

        with open(os.path.join(self.root, "data.bin"), "w") as f:
            f.write("binary data")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_default_crawl(self):
        crawler = FileCrawler(self.root)
        docs = crawler.crawl()
        # Should pick up .md and .py, ignore .bin
        exts = {d["metadata"]["extension"] for d in docs}
        self.assertIn(".md", exts)
        self.assertIn(".py", exts)
        self.assertNotIn(".bin", exts)

    def test_filtered_extensions(self):
        crawler = FileCrawler(self.root, allowed_extensions=[".md"])
        docs = crawler.crawl()
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["metadata"]["extension"], ".md")


if __name__ == "__main__":
    unittest.main()
