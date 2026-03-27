# Copyright (c) 2025
#           Thomas Bömer (thomas.bömer@tu-dortmund.de)
#           Nico Koltermann (nico.koltermann@tu-dortmund.de) 
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from scholarly import scholarly

def fetch_scholar_papers_bib(query, max_results, output_file):
    """
    Fetches Google Scholar papers based on the query and saves them as a BibTeX file.

    Args:
        query (str): The search query string.
        max_results (int): Maximum number of results to fetch.
        output_file (str): File name to save the results in BibTeX format.
    """
    try:
        search_results = scholarly.search_pubs(query)
        results = []

        with open(output_file, "w", encoding="utf-8") as bib_file:
            for i, result in enumerate(search_results):
                if i >= max_results:
                    break

                # Extract BibTeX details
                bib = result.get("bib", {})
                title = bib.get("title", "Unknown Title")
                author = " and ".join(bib.get("author", []))
                year = bib.get("pub_year", "Unknown Year")
                venue = bib.get("venue", "Unknown Venue")
                url = result.get("pub_url", None)
                key = f"{author.split()[0]}_{year}" if author else f"key_{i}"

                # Format BibTeX entry
                bib_entry = f"""
@article{{{key},
  title = {{{title}}},
  author = {{{author}}},
  year = {{{year}}},
  journal = {{{venue}}},
  url = {{{url}}}
}}
"""
                # Write to file
                bib_file.write(bib_entry)
                print(f"Saved entry {i + 1}")

        print(f"Saved all results to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    query = "heuristic pre-marshalling problem"
    max_results = 20
    output_file = "../../../../data/ideas/lit_review_scholar_results_premarshalling.bib"
    fetch_scholar_papers_bib(query, max_results, output_file)
