import scrapy
from scrapy.spiders import CSVFeedSpider
import os
from dmoz.items import DmozItem


class DMozSpider(CSVFeedSpider):
    name = 'dmoz-csv'
    start_urls = ['file:' + str(os.getenv('SITES'))]
    delimiter = '\t'
    quotechar = "'"
    headers = ['domain', 'cat_labels_en']

    def parse_special(self, response, categories, orig_domain):
        item = DmozItem()
        item['categories'] = categories
        item['domain'] = response.url
        item['orig_domain'] = orig_domain
        item['text'] = response.body_as_unicode()
        return item

    def parse_row(self, response, row):
        self.logger.info('Hi, this is a row!: %r', row)
        categories = row['cat_labels_en']
        domain = row['domain']
        return scrapy.Request(url='http://%s' % domain, callback=lambda r: self.parse_special(r, categories, domain))
