
from urllib.parse import urlparse

import sys
import bs4
import requests
import en_core_web_sm
import spacy  as sp
import pandas as pd

sp.prefer_gpu()
nlp = en_core_web_sm.load()


class ScrapTool:
    def visit_url(self, website_url):
        '''
        visit URL
        download the content
        initialize the BeautifulSoup object
        call parsing methods
        return Series object
        '''
        content = requests.get(website_url, timeout=60).content

        # lxml is apparently faster than other settings.
        soup = bs4.BeautifulSoup(content, 'lxml')
        result = { 'website_url'  : website_url
                 , 'website_name' : self.get_website_name(website_url)
                 , 'website_text' : self.get_html_title_tag(soup)
                                  + self.get_html_meta_tags(soup)
                                  + self.get_html_heading_tags(soup)
                                  + self.get_text_content(soup) }

        # convert to Series object
        return pd.Series(result)

    def get_website_name(self, website_url):
        '''
        example: returns "google" from "www.google.com"
        '''
        return ''.join(urlparse(website_url).netloc.split('.')[-2])

    def get_html_title_tag(self, soup):
        '''
        return the text content of <title> tag from a webpage
        '''
        return '. '.join(soup.title.contents)

    def get_html_meta_tags(self, soup):
        '''
        returns the text content of <meta> tags related to keywords and description from a webpage
        '''
        tags = soup.find_all(lambda tag: (tag.name == 'meta') & (tag.has_attr('name') & (tag.has_attr('content'))))
        content = (str(tag['content']) for tag in tags if tag['name'] in ['keywords', 'description'])
        return ' '.join(content)

    def get_html_heading_tags(self, soup):
        '''
        returns the text content of heading tags
        the assumption is that headings might contain relatively important text
        '''
        tags = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        content = (' '.join(tag.stripped_strings) for tag in tags)
        return ' '.join(content)

    def get_text_content(self, soup):
        '''
        returns the text content of the whole page with some exception to tags
        see tags_to_ignore
        '''
        tags_to_ignore = [ 'style', 'script', 'head', 'title', 'meta', '[document]'
                         , 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'noscript' ]
        tags = soup.find_all(text=True)
        result = []
        for tag in tags:
            stripped_tag = tag.strip()
            if tag.parent.name not in tags_to_ignore                        \
                and not isinstance(tag, bs4.element.Comment)                \
                and not stripped_tag.isnumeric()                            \
                and len(stripped_tag) > 0:
                result.append(stripped_tag)
        return ' '.join(result)


def clean_text(document):
    '''
    clean the document
    remove pronouns, stopwords, lemmatize the words and lowercase them
    '''
    doc = nlp(document)

    def clean_token(token) -> str:
        return str(token.lemma_.lower().strip())

    exclusion_list = ['nan']
    def usable_token(token) -> bool:
        return not ( token.is_stop
                  or token.is_punct
                  or token.text.isnumeric()
                  or not token.text.isalnum()
                  or token.text in exclusion_list )

    return ' '.join(clean_token(token)
        for token in doc if usable_token(token))


def die(errmsg: str):
    '''
    print errmsg to stderr and halt the program
    '''
    print(errmsg, file=sys.stderr, flush=True)
    exit(1)

