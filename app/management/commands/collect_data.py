from django.core.management.base import BaseCommand
import time
import pickle
import requests
from bs4 import BeautifulSoup
import MeCab

mecab = MeCab.Tagger('mecabrc')


class Command(BaseCommand):
    def handle(self, *args, **options):
        num_category = 8
        collect_data(num_category)


def extraction_article(url):
    text_list = []
    soup = BeautifulSoup(requests.get(url).content, 'html.parser')
    div = soup.find('div', class_="main article_main")
    for p in div.find_all('p'):
        text = p.string
        text_list.append(str(text))
    return text_list


def extraction_page(url, num, text_lists):
    soup = BeautifulSoup(requests.get(url).content, 'html.parser')
    for div in soup.find_all('div', class_="list_content"):
        title_ = div.find('div', class_="list_title")
        title = title_.string
        a = title_.find('a')
        print(title)
        url_link = a.get("href")
        text_list = extraction_article(url_link)
        text_lists.append(text_list)
    return text_lists


def tokenize(text):
    node = mecab.parseToNode(text)
    time.sleep(1)
    while node:
        if node.feature.split(',')[0] == '名詞':
            yield node.surface.lower()
        node = node.next


def get_words(contents):
    ret = []
    for content in contents:
        ret.append(get_words_main(content))
    return ret


def get_words_main(content):
    return [taken for taken in tokenize(content)]


def collect_data(num):
    """Collect aricles and divide them into words

    Collect aricles from https://gunosy.com/

    Args:
        arg1 (int): number of categories of articles

    Returns:
        None

    """
    print("start collecting data")
    words_list = []
    for i in range(num):
        print("============")
        num_article = i+1
        text_lists = []
        for j in range(5):
            time.sleep(1)
            url = "https://gunosy.com/categories/%s" % str(num_article)
            num_link = j+1
            url += "?page=%s" % str(num_link)
            extraction_page(url, num_link, text_lists)
        print("============")
        if num_article == 1:
            contents_all = text_lists
        else:
            contents_all.extend(text_lists)
    for i in range(len(contents_all)):
        contents_ = [''.join(contents_all[i])]
        words = [' '.join(get_words(contents_)[-1])]
        words_list.append(words[-1])
    with open('./app/python/pickle/words_list.pickle', mode='wb')as f:
        pickle.dump(words_list, f)
    print("finish collecting data")
