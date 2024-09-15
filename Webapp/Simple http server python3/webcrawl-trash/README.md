All of this is to use search in github.com. But there is other problem - how to get stuff from google web search? <br />
https://stackoverflow.com/questions/75400210/how-to-get-full-html-body-of-google-search-using-requests<br />
And this headers looks scary for me. btw. Chatgpt can do a lot but good old stackoverflow and community too. But probably soon it will be competitive for this type of services. If someone can find a way to think conceptually about these types of problems.

```
import urllib.request as urllib2

user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'

url = "https://www.google.com/search?q=Everton+stadium+address"
headers={'User-Agent':user_agent,}

request=urllib2.Request(url,None,headers)
response = urllib2.urlopen(request)
data = response.read()
```

```
import requests

headers = {
    'authority': 'www.google.com',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'accept-language': 'de,de-DE;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6,fr;q=0.5,de-CH;q=0.4,es;q=0.3',
    'cache-control': 'no-cache',
    'dnt': '1',
    'pragma': 'no-cache',
    'sec-ch-ua': '"Not_A Brand";v="99", "Microsoft Edge";v="109", "Chromium";v="109"',
    'sec-ch-ua-arch': '"x86"',
    'sec-ch-ua-bitness': '"64"',
    'sec-ch-ua-full-version': '"109.0.1518.78"',
    'sec-ch-ua-full-version-list': '"Not_A Brand";v="99.0.0.0", "Microsoft Edge";v="109.0.1518.78", "Chromium";v="109.0.5414.120"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-model': '""',
    'sec-ch-ua-platform': '"Windows"',
    'sec-ch-ua-platform-version': '"10.0.0"',
    'sec-ch-ua-wow64': '?0',
    'sec-fetch-dest': 'document',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-site': 'none',
    'sec-fetch-user': '?1',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36 Edg/109.0.1518.78',
}

params = {
    'q': 'Everton stadium address',
}

response = requests.get('https://www.google.com/search', params=params, headers=headers)
print(response.content)
```

<hr>
DEMO 1

```
function crawl(seed_urls, max_depth):
    # Initialize a set to keep track of visited URLs
    visited_urls = set()

    # Initialize a queue for URLs to be crawled
    queue = Queue()

    # Add seed URLs to the queue
    for url in seed_urls:
        queue.enqueue((url, 0))  # (url, depth)

    # Start crawling
    while not queue.is_empty():
        # Dequeue a URL from the queue
        current_url, depth = queue.dequeue()

        # Check if the URL has already been visited
        if current_url in visited_urls:
            continue

        # Visit the URL (e.g., send an HTTP request and retrieve the page content)
        page_content = fetch_page_content(current_url)

        # Parse the page content to extract links
        links = extract_links(page_content)

        # Process the links
        for link in links:
            # Check if the link is within the same domain (optional)
            if is_same_domain(current_url, link):
                # Add the link to the queue with increased depth
                if depth + 1 <= max_depth:
                    queue.enqueue((link, depth + 1))

        # Mark the current URL as visited
        visited_urls.add(current_url)

        # Process the page content (e.g., index the content for search)
        index_page_content(current_url, page_content)

    # Return the set of visited URLs
    return visited_urls

# Example function for fetching page content (simplified)
function fetch_page_content(url):
    # Send an HTTP request to the URL and retrieve the page content
    # (implementation details omitted)
    return page_content

# Example function for extracting links from page content (simplified)
function extract_links(page_content):
    # Use a regular expression or HTML parser to extract links from the page content
    # (implementation details omitted)
    return links

# Example function for checking if two URLs are within the same domain (simplified)
function is_same_domain(url1, url2):
    # Extract the domain from the URLs and compare
    # (implementation details omitted)
    return True if domain(url1) == domain(url2) else False

# Example function for indexing page content (simplified)
function index_page_content(url, page_content):
    # Index the page content for search (e.g., store it in a database)
    # (implementation details omitted)
    pass
```
DEMO 2
```
class WebPage:
    def __init__(self, url, title, content, authority_score):
        self.url = url
        self.title = title
        self.content = content
        self.authority_score = authority_score

class SearchQuery:
    def __init__(self, keywords):
        self.keywords = keywords

class SearchResult:
    def __init__(self, webpage, relevance_score):
        self.webpage = webpage
        self.relevance_score = relevance_score

class RankingAlgorithm:
    def __init__(self, webpages):
        self.webpages = webpages

    def rank_results(self, search_query):
        results = []
        for webpage in self.webpages:
            relevance_score = self.calculate_relevance(webpage, search_query)
            results.append(SearchResult(webpage, relevance_score))
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results

    def calculate_relevance(self, webpage, search_query):
        # Calculate relevance based on keyword match
        relevance_score = self.calculate_keyword_relevance(webpage, search_query)

        # Adjust relevance based on authority score
        relevance_score *= webpage.authority_score

        return relevance_score

    def calculate_keyword_relevance(self, webpage, search_query):
        # Simple keyword matching: count the number of times keywords appear in the content and title
        content_score = sum(webpage.content.lower().count(keyword.lower()) for keyword in search_query.keywords)
        title_score = sum(webpage.title.lower().count(keyword.lower()) for keyword in search_query.keywords)
        
        # Combine content and title scores
        total_score = content_score + title_score

        return total_score

# Example usage
webpages = [
    WebPage("https://example.com/page1", "Page 1 Title", "This is the content of page 1.", 0.8),
    WebPage("https://example.com/page2", "Page 2 Title", "Page 2 contains relevant information.", 0.6),
    # Add more webpages...
]

ranking_algorithm = RankingAlgorithm(webpages)

search_query = SearchQuery(["relevant", "information"])
search_results = ranking_algorithm.rank_results(search_query)

# Display search results
for index, result in enumerate(search_results, start=1):
    print(f"{index}. {result.webpage.title} - {result.webpage.url} - Relevance Score: {result.relevance_score}")
```
DEMO 3
```
pip install requests beautifulsoup4
```
```
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque

class SimpleWebCrawler:
    def __init__(self, base_url, keyword, max_depth=2):
        self.base_url = base_url
        self.keyword = keyword
        self.max_depth = max_depth
        self.visited_urls = set()
        self.urls_to_crawl = deque([(base_url, 0)])
        self.found_links = []

    def crawl(self):
        while self.urls_to_crawl:
            current_url, depth = self.urls_to_crawl.popleft()
            if current_url in self.visited_urls or depth > self.max_depth:
                continue

            self.visited_urls.add(current_url)
            try:
                response = requests.get(current_url)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    self.search_links(soup, current_url)
                    self.enqueue_links(soup, current_url, depth)
            except requests.RequestException as e:
                print(f"Failed to fetch {current_url}: {e}")

    def search_links(self, soup, current_url):
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if self.keyword.lower() in href.lower():
                absolute_url = urljoin(current_url, href)
                if self.is_same_domain(absolute_url):
                    self.found_links.append(absolute_url)
                    print(f"Found link: {absolute_url}")

    def enqueue_links(self, soup, current_url, depth):
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            absolute_url = urljoin(current_url, href)
            if self.is_same_domain(absolute_url) and absolute_url not in self.visited_urls:
                self.urls_to_crawl.append((absolute_url, depth + 1))

    def is_same_domain(self, url):
        return urlparse(url).netloc == urlparse(self.base_url).netloc

# Usage example
if __name__ == "__main__":
    base_url = "https://example.com"  # Replace with the target site URL
    keyword = "contact"  # Replace with the keyword you are searching for
    crawler = SimpleWebCrawler(base_url, keyword)
    crawler.crawl()

    print("\nFound links containing the keyword:")
    for link in crawler.found_links:
        print(link)
```
DEMO 4
```
pip install requests beautifulsoup4
```
```
import requests
from bs4 import BeautifulSoup

class GoogleSearchCrawler:
    def __init__(self, keyword, num_results=10):
        self.keyword = keyword
        self.num_results = num_results
        self.search_url = "https://www.google.com/search"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def search(self):
        params = {"q": self.keyword, "num": self.num_results}
        response = requests.get(self.search_url, headers=self.headers, params=params)
        if response.status_code == 200:
            return response.text
        else:
            print(f"Failed to fetch search results: {response.status_code}")
            return None

    def parse_results(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        results = []
        for g in soup.find_all('div', class_='g'):
            anchors = g.find_all('a')
            if anchors:
                link = anchors[0]['href']
                title = g.find('h3').text if g.find('h3') else 'No title'
                results.append({'title': title, 'link': link})
        return results

    def display_results(self, results):
        for index, result in enumerate(results, start=1):
            print(f"{index}. {result['title']}\n   {result['link']}\n")

# Usage example
if __name__ == "__main__":
    keyword = "python programming"  # Replace with the keyword you are searching for
    crawler = GoogleSearchCrawler(keyword)
    html = crawler.search()
    if html:
        results = crawler.parse_results(html)
        crawler.display_results(results)
```
DEMO 5
```
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

class GoogleSearchCrawler:
    def __init__(self, keyword, num_results=10):
        self.keyword = keyword
        self.num_results = num_results
        self.search_url = "https://www.google.com/search"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def search(self):
        params = {"q": self.keyword, "num": self.num_results}
        response = requests.get(self.search_url, headers=self.headers, params=params)
        if response.status_code == 200:
            return response.text
        else:
            print(f"Failed to fetch search results: {response.status_code}")
            return None

    def parse_results(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        results = []
        for g in soup.find_all('div', class_='g'):
            anchors = g.find_all('a')
            if anchors:
                link = anchors[0]['href']
                title = g.find('h3').text if g.find('h3') else 'No title'
                results.append({'title': title, 'link': link})
        return results

    def search_in_page(self, url, search_list):
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                page_content = response.text
                soup = BeautifulSoup(page_content, 'html.parser')
                found_items = {item: False for item in search_list}
                text = soup.get_text().lower()
                for item in search_list:
                    if item.lower() in text:
                        found_items[item] = True
                return found_items
            else:
                print(f"Failed to fetch page content: {url} - Status code: {response.status_code}")
                return None
        except requests.RequestException as e:
            print(f"Failed to fetch page content: {url} - Exception: {e}")
            return None

    def display_results(self, results):
        for index, result in enumerate(results, start=1):
            print(f"{index}. {result['title']}\n   {result['link']}\n")

    def run(self, search_list):
        html = self.search()
        if html:
            results = self.parse_results(html)
            self.display_results(results)

            for result in results:
                print(f"\nSearching in page: {result['link']}")
                found_items = self.search_in_page(result['link'], search_list)
                if found_items:
                    print(f"Results for {result['link']}:")
                    for item, found in found_items.items():
                        status = "Found" if found else "Not Found"
                        print(f"  {item}: {status}")
                print("\n")

# Usage example
if __name__ == "__main__":
    keyword = "python programming"  # Replace with the keyword you are searching for
    search_list = ["installation", "syntax", "variables", "loops", "functions"]  # Replace with the list of items to search for
    crawler = GoogleSearchCrawler(keyword)
    crawler.run(search_list)
```
DEMO 6
```
import requests
from bs4 import BeautifulSoup

class GitHubSearchCrawler:
    def __init__(self, keyword, num_results=10):
        self.keyword = keyword
        self.num_results = num_results
        self.search_url = "https://github.com/search"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def search(self):
        params = {"q": self.keyword, "type": "repositories"}
        response = requests.get(self.search_url, headers=self.headers, params=params)
        if response.status_code == 200:
            return response.text
        else:
            print(f"Failed to fetch search results: {response.status_code}")
            return None

    def parse_results(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        results = []
        repo_list = soup.find_all('li', class_='repo-list-item')
        for repo in repo_list[:self.num_results]:
            title_tag = repo.find('a', class_='v-align-middle')
            description_tag = repo.find('p', class_='mb-1')
            stars_tag = repo.find('a', class_='Link--muted')
            title = title_tag.text.strip() if title_tag else 'No title'
            link = f"https://github.com{title_tag['href']}" if title_tag else 'No link'
            description = description_tag.text.strip() if description_tag else 'No description'
            stars = stars_tag.text.strip() if stars_tag else '0'
            results.append({'title': title, 'link': link, 'description': description, 'stars': stars})
        return results

    def display_results(self, results):
        for index, result in enumerate(results, start=1):
            print(f"{index}. {result['title']} - Stars: {result['stars']}")
            print(f"   {result['link']}")
            print(f"   {result['description']}\n")

# Usage example
if __name__ == "__main__":
    keyword = "renderer"  # Replace with the keyword you are searching for
    crawler = GitHubSearchCrawler(keyword)
    html = crawler.search()
    if html:
        results = crawler.parse_results(html)
        crawler.display_results(results)
```
DEMO 7
```
import requests
from bs4 import BeautifulSoup

class GitHubSearchCrawler:
    def __init__(self, keyword, num_results=10):
        self.keyword = keyword
        self.num_results = num_results
        self.search_url = "https://github.com/search"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def search(self):
        params = {"q": self.keyword, "type": "repositories"}
        response = requests.get(self.search_url, headers=self.headers, params=params)
        if response.status_code == 200:
            return response.text
        else:
            print(f"Failed to fetch search results: {response.status_code}")
            return None

    def parse_results(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        results = []
        repo_list = soup.find_all('div', class_='repo-list-item')
        for repo in repo_list[:self.num_results]:
            title_tag = repo.find('a', class_='v-align-middle')
            description_tag = repo.find('p', class_='mb-1')
            stars_tag = repo.find('a', class_='Link--muted')
            title = title_tag.text.strip() if title_tag else 'No title'
            link = f"https://github.com{title_tag['href']}" if title_tag else 'No link'
            description = description_tag.text.strip() if description_tag else 'No description'
            stars = stars_tag.text.strip() if stars_tag else '0'
            results.append({'title': title, 'link': link, 'description': description, 'stars': stars})
        return results

    def display_results(self, results):
        for index, result in enumerate(results, start=1):
            print(f"{index}. {result['title']} - Stars: {result['stars']}")
            print(f"   {result['link']}")
            print(f"   {result['description']}\n")

# Usage example
if __name__ == "__main__":
    keyword = "renderer"  # Replace with the keyword you are searching for
    crawler = GitHubSearchCrawler(keyword)
    html = crawler.search()
    if html:
        results = crawler.parse_results(html)
        crawler.display_results(results)
```
DEMO 8
```
import requests
from bs4 import BeautifulSoup

class GitHubSearchCrawler:
    def __init__(self, keyword, num_results=10):
        self.keyword = keyword
        self.num_results = num_results
        self.search_url = "https://github.com/search"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def search(self):
        params = {"q": self.keyword, "type": "repositories"}
        response = requests.get(self.search_url, headers=self.headers, params=params)
        if response.status_code == 200:
            return response.text
        else:
            print(f"Failed to fetch search results: {response.status_code}")
            return None

    def parse_results(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        results = []
        repo_list = soup.find_all('li', class_='repo-list-item')
        for repo in repo_list[:self.num_results]:
            title_tag = repo.find('a', class_='v-align-middle')
            description_tag = repo.find('p', class_='mb-1')
            stars_tag = repo.find('a', class_='Link--muted')
            title = title_tag.text.strip() if title_tag else 'No title'
            link = f"https://github.com{title_tag['href']}" if title_tag else 'No link'
            description = description_tag.text.strip() if description_tag else 'No description'
            stars = stars_tag.text.strip() if stars_tag else '0'
            results.append({'title': title, 'link': link, 'description': description, 'stars': stars})
        return results

    def display_results(self, results):
        for index, result in enumerate(results, start=1):
            print(f"{index}. {result['title']} - Stars: {result['stars']}")
            print(f"   {result['link']}")
            print(f"   {result['description']}\n")

# Usage example
if __name__ == "__main__":
    keyword = "renderer"  # Replace with the keyword you are searching for
    crawler = GitHubSearchCrawler(keyword)
    html = crawler.search()
    if html:
        results = crawler.parse_results(html)
        crawler.display_results(results)
```
DEMO 9
```
import requests
from bs4 import BeautifulSoup

class GitHubSearchCrawler:
    def __init__(self, keyword, num_results=10):
        self.keyword = keyword
        self.num_results = num_results
        self.search_url = "https://github.com/search"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def search(self):
        params = {"q": self.keyword, "type": "repositories"}
        try:
            response = requests.get(self.search_url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Failed to fetch search results: {e}")
            return None

    def parse_results(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        results = []
        repo_list = soup.find_all('li', class_='repo-list-item')
        if not repo_list:
            print("No repositories found. Check the HTML structure or the query parameters.")
        for repo in repo_list[:self.num_results]:
            title_tag = repo.find('a', class_='v-align-middle')
            description_tag = repo.find('p', class_='mb-1')
            stars_tag = repo.find('a', class_='Link--muted')
            title = title_tag.text.strip() if title_tag else 'No title'
            link = f"https://github.com{title_tag['href']}" if title_tag else 'No link'
            description = description_tag.text.strip() if description_tag else 'No description'
            stars = stars_tag.text.strip() if stars_tag else '0'
            results.append({'title': title, 'link': link, 'description': description, 'stars': stars})
        return results

    def display_results(self, results):
        for index, result in enumerate(results, start=1):
            print(f"{index}. {result['title']} - Stars: {result['stars']}")
            print(f"   {result['link']}")
            print(f"   {result['description']}\n")

# Usage example
if __name__ == "__main__":
    keyword = "renderer"  # Replace with the keyword you are searching for
    crawler = GitHubSearchCrawler(keyword)
    html = crawler.search()
    if html:
        results = crawler.parse_results(html)
        crawler.display_results(results)
    else:
        print("Failed to retrieve or parse search results.")
```
DEMO 10
```
import requests
from bs4 import BeautifulSoup

class GitHubSearchCrawler:
    def __init__(self, keyword, num_results=10):
        self.keyword = keyword
        self.num_results = num_results
        self.search_url = "https://github.com/search"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def search(self):
        params = {"q": self.keyword, "type": "repositories"}
        try:
            response = requests.get(self.search_url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Failed to fetch search results: {e}")
            return None

    def parse_results(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        results = []
        repo_list = soup.find_all('div', class_='f4 text-normal')
        if not repo_list:
            print("No repositories found. Check the HTML structure or the query parameters.")
        for repo in repo_list[:self.num_results]:
            title_tag = repo.find('a')
            description_tag = repo.find_next('p', class_='mb-1')
            stars_tag = repo.find_next('a', class_='Link--muted')
            title = title_tag.text.strip() if title_tag else 'No title'
            link = f"https://github.com{title_tag['href']}" if title_tag else 'No link'
            description = description_tag.text.strip() if description_tag else 'No description'
            stars = stars_tag.text.strip() if stars_tag else '0'
            results.append({'title': title, 'link': link, 'description': description, 'stars': stars})
        return results

    def display_results(self, results):
        for index, result in enumerate(results, start=1):
            print(f"{index}. {result['title']} - Stars: {result['stars']}")
            print(f"   {result['link']}")
            print(f"   {result['description']}\n")

# Usage example
if __name__ == "__main__":
    keyword = "renderer"  # Replace with the keyword you are searching for
    crawler = GitHubSearchCrawler(keyword)
    html = crawler.search()
    if html:
        results = crawler.parse_results(html)
        crawler.display_results(results)
    else:
        print("Failed to retrieve or parse search results.")
```
DEMO 11
```
import requests
from bs4 import BeautifulSoup

class GitHubSearchCrawler:
    def __init__(self, keyword, num_results=10):
        self.keyword = keyword
        self.num_results = num_results
        self.search_url = "https://github.com/search"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def search(self, search_type):
        params = {"q": self.keyword, "type": search_type}
        try:
            response = requests.get(self.search_url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Failed to fetch search results: {e}")
            return None

    def parse_repository_results(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        results = []
        repo_list = soup.find_all('div', class_='f4 text-normal')
        if not repo_list:
            print("No repositories found. Check the HTML structure or the query parameters.")
        for repo in repo_list[:self.num_results]:
            title_tag = repo.find('a')
            description_tag = repo.find_next('p', class_='mb-1')
            stars_tag = repo.find_next('a', class_='Link--muted')
            title = title_tag.text.strip() if title_tag else 'No title'
            link = f"https://github.com{title_tag['href']}" if title_tag else 'No link'
            description = description_tag.text.strip() if description_tag else 'No description'
            stars = stars_tag.text.strip() if stars_tag else '0'
            results.append({'title': title, 'link': link, 'description': description, 'stars': stars})
        return results

    def parse_topic_results(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        results = []
        topic_list = soup.find_all('a', class_='topic-tag topic-tag-link f6 my-1')
        if not topic_list:
            print("No topics found. Check the HTML structure or the query parameters.")
        for topic in topic_list[:self.num_results]:
            topic_name = topic.text.strip()
            link = f"https://github.com{topic['href']}" if topic else 'No link'
            results.append({'topic': topic_name, 'link': link})
        return results

    def display_repository_results(self, results):
        print("Repository Results:\n")
        for index, result in enumerate(results, start=1):
            print(f"{index}. {result['title']} - Stars: {result['stars']}")
            print(f"   {result['link']}")
            print(f"   {result['description']}\n")

    def display_topic_results(self, results):
        print("Topic Results:\n")
        for index, result in enumerate(results, start=1):
            print(f"{index}. {result['topic']}")
            print(f"   {result['link']}\n")

# Usage example
if __name__ == "__main__":
    keyword = "renderer"  # Replace with the keyword you are searching for
    crawler = GitHubSearchCrawler(keyword)

    # Search and display repositories
    repo_html = crawler.search("repositories")
    if repo_html:
        repo_results = crawler.parse_repository_results(repo_html)
        crawler.display_repository_results(repo_results)

    # Search and display topics
    topic_html = crawler.search("topics")
    if topic_html:
        topic_results = crawler.parse_topic_results(topic_html)
        crawler.display_topic_results(topic_results)
    else:
        print("Failed to retrieve or parse search results.")
```
DEMO 12
```
import requests
from bs4 import BeautifulSoup
import urllib.parse

class GitHubSearchCrawler:
    def __init__(self, keyword, num_results=10):
        self.keyword = keyword
        self.num_results = num_results
        self.search_url = "https://github.com/search"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def build_url(self, search_type):
        params = {"q": self.keyword, "type": search_type}
        url = f"{self.search_url}?{urllib.parse.urlencode(params)}"
        return url

    def search(self, search_type):
        url = self.build_url(search_type)
        print(f"Fetching URL: {url}")  # Print the URL being fetched
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Failed to fetch search results: {e}")
            return None

    def parse_repository_results(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        results = []
        repo_list = soup.find_all('div', class_='f4 text-normal')
        if not repo_list:
            print("No repositories found. Check the HTML structure or the query parameters.")
        for repo in repo_list[:self.num_results]:
            title_tag = repo.find('a')
            description_tag = repo.find_next('p', class_='mb-1')
            stars_tag = repo.find_next('a', class_='Link--muted')
            title = title_tag.text.strip() if title_tag else 'No title'
            link = f"https://github.com{title_tag['href']}" if title_tag else 'No link'
            description = description_tag.text.strip() if description_tag else 'No description'
            stars = stars_tag.text.strip() if stars_tag else '0'
            results.append({'title': title, 'link': link, 'description': description, 'stars': stars})
        return results

    def parse_topic_results(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        results = []
        topic_list = soup.find_all('a', class_='topic-tag topic-tag-link f6 my-1')
        if not topic_list:
            print("No topics found. Check the HTML structure or the query parameters.")
        for topic in topic_list[:self.num_results]:
            topic_name = topic.text.strip()
            link = f"https://github.com{topic['href']}" if topic else 'No link'
            results.append({'topic': topic_name, 'link': link})
        return results

    def display_repository_results(self, results):
        print("Repository Results:\n")
        for index, result in enumerate(results, start=1):
            print(f"{index}. {result['title']} - Stars: {result['stars']}")
            print(f"   {result['link']}")
            print(f"   {result['description']}\n")

    def display_topic_results(self, results):
        print("Topic Results:\n")
        for index, result in enumerate(results, start=1):
            print(f"{index}. {result['topic']}")
            print(f"   {result['link']}\n")

# Usage example
if __name__ == "__main__":
    keyword = "renderer"  # Replace with the keyword you are searching for
    crawler = GitHubSearchCrawler(keyword)

    # Search and display repositories
    repo_html = crawler.search("repositories")
    if repo_html:
        repo_results = crawler.parse_repository_results(repo_html)
        crawler.display_repository_results(repo_results)

    # Search and display topics
    topic_html = crawler.search("topics")
    if topic_html:
        topic_results = crawler.parse_topic_results(topic_html)
        crawler.display_topic_results(topic_results)
    else:
        print("Failed to retrieve or parse search results.")
```
DEMO 13
```
import requests
import json

class GitHubSearchCrawler:
    def __init__(self, keyword, num_results=10):
        self.keyword = keyword
        self.num_results = num_results
        self.search_url = "https://api.github.com/search/repositories"
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def build_url(self):
        params = {
            "q": self.keyword,
            "sort": "stars",
            "order": "desc",
            "per_page": self.num_results
        }
        url = f"{self.search_url}?{urllib.parse.urlencode(params)}"
        return url

    def search(self):
        url = self.build_url()
        print(f"Fetching URL: {url}")  # Print the URL being fetched
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Failed to fetch search results: {e}")
            return None

    def parse_results(self, data):
        results = []
        items = data.get('items', [])
        for item in items:
            repo_info = {
                'name': item['name'],
                'full_name': item['full_name'],
                'html_url': item['html_url'],
                'description': item['description'],
                'stargazers_count': item['stargazers_count'],
                'language': item['language'],
                'topics': item.get('topics', []),
                'owner': {
                    'login': item['owner']['login'],
                    'html_url': item['owner']['html_url']
                }
            }
            results.append(repo_info)
        return results

    def display_results(self, results):
        print("Repository Results:\n")
        for index, result in enumerate(results, start=1):
            print(f"{index}. {result['full_name']} - Stars: {result['stargazers_count']}")
            print(f"   URL: {result['html_url']}")
            print(f"   Description: {result['description']}")
            print(f"   Language: {result['language']}")
            print(f"   Topics: {', '.join(result['topics']) if result['topics'] else 'None'}")
            print(f"   Owner: {result['owner']['login']} ({result['owner']['html_url']})\n")

# Usage example
if __name__ == "__main__":
    keyword = "renderer"  # Replace with the keyword you are searching for
    crawler = GitHubSearchCrawler(keyword)

    # Search and display repositories
    data = crawler.search()
    if data:
        results = crawler.parse_results(data)
        crawler.display_results(results)
    else:
        print("Failed to retrieve or parse search results.")
```
DEMO 14
```
import requests
from collections import Counter
from bs4 import BeautifulSoup
import re

class GitHubRepositoryResearcher:
    def __init__(self, repo_url):
        self.repo_url = repo_url
        self.api_url = self.construct_api_url(repo_url)
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def construct_api_url(self, repo_url):
        repo_path = repo_url.replace("https://github.com/", "")
        return f"https://api.github.com/repos/{repo_path}/contents"

    def fetch_repository_content(self):
        try:
            response = requests.get(self.api_url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Failed to fetch repository content: {e}")
            return None

    def extract_links(self, html_content):
        soup = BeautifulSoup(html_content, "html.parser")
        links = [a['href'] for a in soup.find_all('a', href=True)]
        return links

    def find_most_frequent_word(self, text):
        words = re.findall(r'\b\w+\b', text.lower())
        word_counts = Counter(words)
        most_common_word, count = word_counts.most_common(1)[0]
        return most_common_word, count

    def research_repository(self):
        content_data = self.fetch_repository_content()
        if not content_data:
            return

        all_text = ""
        all_links = []

        for file_info in content_data:
            if file_info['type'] == 'file':
                file_url = file_info['download_url']
                try:
                    file_response = requests.get(file_url)
                    file_response.raise_for_status()
                    file_content = file_response.text
                    all_text += file_content

                    # Extract links from HTML files
                    if file_info['name'].endswith(".html"):
                        links = self.extract_links(file_content)
                        all_links.extend(links)
                except requests.RequestException as e:
                    print(f"Failed to fetch file content: {e}")

        most_common_word, count = self.find_most_frequent_word(all_text)
        print("Most frequently used word:", most_common_word)
        print("Count:", count)
        print("Links found in HTML files:", all_links)

# Usage example
if __name__ == "__main__":
    repo_url = "https://github.com/adam-p/markdown-here"
    researcher = GitHubRepositoryResearcher(repo_url)
    researcher.research_repository()
```
DEMO 15
```
import requests
from collections import Counter
import re

class GitHubFileAnalyzer:
    def __init__(self, file_url):
        self.file_url = file_url
        self.raw_url = self.construct_raw_url(file_url)
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def construct_raw_url(self, file_url):
        raw_url = file_url.replace("github.com", "raw.githubusercontent.com")
        raw_url = raw_url.replace("/blob/", "/")
        return raw_url

    def fetch_file_content(self):
        try:
            response = requests.get(self.raw_url, headers=self.headers)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Failed to fetch file content: {e}")
            return None

    def find_most_frequent_word(self, text):
        words = re.findall(r'\b\w+\b', text.lower())
        word_counts = Counter(words)
        most_common_word, count = word_counts.most_common(1)[0]
        return most_common_word, count

    def analyze_file(self):
        file_content = self.fetch_file_content()
        if not file_content:
            return

        most_common_word, count = self.find_most_frequent_word(file_content)
        print("Most frequently used word:", most_common_word)
        print("Count:", count)

# Usage example
if __name__ == "__main__":
    file_url = "https://github.com/gordicaleksa/slovenian-llm-eval/blob/slovenian_eval_translate/lm_eval/models/huggingface.py"
    analyzer = GitHubFileAnalyzer(file_url)
    analyzer.analyze_file()
```
DEMO 16
```
import requests
from collections import Counter, defaultdict
import re

class GitHubFileAnalyzer:
    def __init__(self, file_url):
        self.file_url = file_url
        self.raw_url = self.construct_raw_url(file_url)
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def construct_raw_url(self, file_url):
        raw_url = file_url.replace("github.com", "raw.githubusercontent.com")
        raw_url = raw_url.replace("/blob/", "/")
        return raw_url

    def fetch_file_content(self):
        try:
            response = requests.get(self.raw_url, headers=self.headers)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Failed to fetch file content: {e}")
            return None

    def count_words_and_lines(self, text):
        words_counter = Counter()
        word_lines = defaultdict(list)
        lines = text.split('\n')
        
        for line_number, line in enumerate(lines, start=1):
            words = re.findall(r'\b\w+\b', line.lower())
            words_counter.update(words)
            for word in words:
                word_lines[word].append(line_number)

        return words_counter, word_lines

    def analyze_file(self):
        file_content = self.fetch_file_content()
        if not file_content:
            return

        words_counter, word_lines = self.count_words_and_lines(file_content)
        
        print("All words with their counts and line numbers:")
        for word, count in words_counter.items():
            lines = ', '.join(map(str, word_lines[word]))
            print(f"Word: '{word}' - Count: {count} - Lines: {lines}")

# Usage example
if __name__ == "__main__":
    file_url = "https://github.com/gordicaleksa/slovenian-llm-eval/blob/slovenian_eval_translate/lm_eval/models/huggingface.py"
    analyzer = GitHubFileAnalyzer(file_url)
    analyzer.analyze_file()
```
