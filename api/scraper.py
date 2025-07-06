import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time

def scrape_data(company="ultrajaya"):
    options = Options()
    options.add_argument("--headless")  # Tanpa tampilan browser

    driver = webdriver.Chrome(options=options)
    driver.get(f"https://search.katadata.co.id/search?q={company}&source=databoks")

    # Tunggu konten termuat (delay JS)
    time.sleep(2)  # Bisa diganti dengan WebDriverWait
    
    
    # Ambil konten HTML setelah JS selesai render
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    results = soup.find_all('p')
    
    companyInfo = []
    if len(results) > 0:
    # Lanjutkan parsing seperti biasa
        for item in results[:-2]:
            companyInfo.append(item.get_text(strip=True))

    driver.quit()
    # url = f"https://id.kompass.com/a/{keyword}/{location}/"


    # companies = []
    # Contoh: scrapping judul + link + cuplikan (snippet)
    # results = soup.find_all('p')


    # for item in results:
         # Dapatkan cuplikan/deskripsi
        # snippet_tag = item.find('div', class_='text-sm leading-relaxed space-y-3 ')
        # snippet = snippet_tag.get_text(strip=True) if snippet_tag else 'N/A'
    # cards = soup.select("text-sm leading-relaxed space-y-3")
    
    # for card in cards:
    #     name = card.select_one(".card__name")
    #     address = card.select_one(".card__address")
    #     activity = card.select_one(".card__activity")

    #     companies.append({
    #         "name": name.text.strip() if name else "N/A",
    #         "address": address.text.strip() if address else "N/A",
    #         "sector": activity.text.strip() if activity else "N/A"
    #     })

    return companyInfo
