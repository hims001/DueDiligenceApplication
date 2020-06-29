from django.test import TestCase
from selenium import webdriver
import time


class FunctionalTestCase(TestCase):

    def setUp(self):
        self.browser = webdriver.Chrome()
        self.browser.get('http://localhost:8000')

    def tearDown(self):
        self.browser.quit()

    def test_homepage_loaded(self):
        self.assertIn('Due Diligence Portal', self.browser.page_source)

    def test_search_successful(self):
        textbox = self.browser.find_element_by_id('id_SearchText')
        textbox.send_keys('ABC Pvt. Ltd.')
        self.browser.find_element_by_id('btnSubmit').click()
        time.sleep(5)
        self.assertIn('Sentiment', self.browser.page_source)