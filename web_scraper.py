import requests
from bs4 import BeautifulSoup
import pandas as pd
from named_entity_extraction.Config.config import path, new_atlas_urls, headers, imnovation_urls, sciencenews_urls
import abc


class webscraper(metaclass=abc.ABCMeta):
    def __init__(self):
        self.path = path
        self.headers = headers

    @abc.abstractmethod
    def run(self):
        pass


class new_atlas(webscraper):
    def __init__(self):
        super().__init__()
        self.urls = new_atlas_urls

    def run(self):

        final_df = pd.DataFrame(columns=["Category", "Pub_Date", "Title"])
        for url in self.urls:
            pg_num = 1
            while pg_num < 100:

                title = []
                category = []
                pub_date = []
                response = requests.get(url.format(pg_num), headers=self.headers)

                soup = BeautifulSoup(response.content, "html.parser")

                categorys = soup.find("body").find_all(class_="PromoB-category")
                pub_dates = soup.find("body").find_all(class_="PromoB-date")
                titles = soup.find('body').find_all('h3')

                for x in titles:
                    title.append(x.text.strip())
                for x in categorys:
                    category.append(x.text.strip())
                for x in pub_dates:
                    pub_date.append(x.text.strip())

                data = {'Category': category,
                        'Pub_Date': pub_date,
                        'Title': title}
                df_temp = pd.DataFrame(data)
                final_df = pd.concat([final_df, df_temp], ignore_index=True, axis=0)
                pg_num += 1
                print(pg_num)
            # print(f"""Sucessfully extracted for {categorys[0].text.strip()}""")
            final_df.to_csv(f"""{path}new_atlas2.csv""")


class imnovation(webscraper):

    def __init__(self):
        super().__init__()
        self.urls = imnovation_urls

    def run(self):

        final_df = pd.DataFrame(columns=["Category", "Title"])
        for url in self.urls:
            pg_num = 1
            while (pg_num < 15):
                try:

                    title = []
                    category = []

                    response = requests.get(url.format(pg_num), headers=self.headers)

                    soup = BeautifulSoup(response.content, "html.parser")

                    categorys = soup.find(class_="normal module-distributor-grid").find_all(
                        class_="module_distributor-label text08")
                    titles = soup.find(class_="normal module-distributor-grid").find_all(
                        class_="module_distributor-title text04")

                    for x in titles:
                        title.append(x.text.strip())
                    for x in categorys:
                        category.append(x.text.strip())

                    data = {'Category': category,
                            'Title': title}
                    df_temp = pd.DataFrame(data)
                    final_df = pd.concat([final_df, df_temp], ignore_index=True, axis=0)
                    pg_num += 1
                    print(pg_num)
                except:
                    pass
            final_df.to_csv(f"""{path}imnovation.csv""")


class sciencenews():
    def __init__(self):
        super().__init__()
        self.urls = sciencenews_urls

    def run(self):

        final_df = pd.DataFrame(columns=["Category", "Title"])
        for url in self.urls:
            pg_num = 1
            while (pg_num < 100):
                try:

                    title = []
                    category = []

                    response = requests.get(url.format(pg_num), headers=self.headers)

                    soup = BeautifulSoup(response.content, "html.parser")

                    categorys = soup.find(class_="river-with-sidebar__list___1EfmS").find_all(
                        class_="post-item-river__eyebrow___33ASW")
                    titles = soup.find(class_="river-with-sidebar__list___1EfmS").find_all(
                        class_="post-item-river__title___J3spU")

                    for x in titles:
                        title.append(x.text.strip())
                    for x in categorys:
                        category.append(x.text.strip())

                    data = {'Category': category,
                            'Title': title}
                    df_temp = pd.DataFrame(data)
                    final_df = pd.concat([final_df, df_temp], ignore_index=True, axis=0)
                    pg_num += 1
                    print(pg_num)
                except:
                    pass
            final_df.to_csv("./sample_files/sciencenews.csv")


if __name__ == "__main__":
    new_atlas = new_atlas()
    new_atlas.run()

    imnovation = imnovation()
    imnovation.run()

    sciencenews = sciencenews()
    sciencenews.run()
