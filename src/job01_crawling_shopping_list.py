#job01_crawling_shopping_title
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import datetime
from selenium import webdriver as wb  # 동적 사이트 수집
import time
from selenium.webdriver.common.action_chains import ActionChains

# webdriver사용 세팅
options = ChromeOptions()
user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
options.add_argument('user_agent=' + user_agent)
options.add_argument('lang=ko_KR')

service = ChromeService(executable_path=ChromeDriverManager().install())
driver = wb.Chrome(service=service, options=options)

category = ['travel', 'Food', '', '', 'Fashion', '', '', '', 'Beauty', '', 'Fu/In', '', 'HA/Di']
df_titles = pd.DataFrame()
re_title = re.compile('[^가-힣|a-z|A-Z]')
section_index_num = [4, 8, 10, 12, 1]
for i in section_index_num:
    url = 'https://www.11st.co.kr/browsing/BestSeller.tmall?method=getBestSellerMain&cornerNo={}#pageNum%%13'.format(i)
    driver.get(url)
    time.sleep(1)
    title_tags = driver.find_elements(By.CLASS_NAME,'pname')
    titles = []
    for title_tag in title_tags:
        titles.append(re_title.sub(' ', title_tag.text))

    df_section_titles = pd.DataFrame(titles, columns=['titles'])
    df_section_titles['category'] = category[i]
    df_titles = pd.concat([df_titles, df_section_titles], axis='rows', ignore_index=True)

url = 'https://www.11st.co.kr/browsing/BestSeller.tmall?method=getBestSellerMain&cornerNo=13&dispCtgrNo=2878#pageNum%%13'
driver.get(url)
time.sleep(1)
title_tags = driver.find_elements(By.CLASS_NAME, 'pname')
titles = []

#travel 따로 크롤링
for title_tag in title_tags:
    titles.append(re_title.sub(' ', title_tag.text))

df_section_titles = pd.DataFrame(titles, columns=['titles'])
df_section_titles['category'] = category[0]
df_titles = pd.concat([df_titles, df_section_titles], axis='rows', ignore_index=True)

print(df_titles.head())
df_titles.info()
print(df_titles['category'].value_counts())
df_titles.to_csv('./crawling_data/shopping_title{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d')), index=False)



search_box = driver.find_element('xpath',
                                  '//*[@id="gnbCategory"]/div/div[1]/div[2]/nav/ul/li[1]/a')
actions = wb.ActionChains(driver).move_to_element(search_box)
actions.perform()
