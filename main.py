"""
Hi there! This is the central file for all processes and flows used to grab, clean, and model the sweet data provided from WRDS database
"""
#Packages for data manipulation
import pandas as pd
import numpy as np
import scipy.stats as stats
from collections import defaultdict
import json
from datetime import datetime

#Packages for API calls and database connection
import requests as r
from bs4 import BeautifulSoup
import time
from sqlalchemy import create_engine
from datetime import datetime, timedelta

#NLP Packages
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import *
from fuzzywuzzy import fuzz

#Misc packages
from dotenv import load_dotenv
import os

class Main():

    def __init__(self):
        self.engine = create_engine(os.environ.get("DB_URI"))

    def get_fin_data(self, file_path: str):
        """
        Parameters:
            - file_path: the path to the file that holds the data
        Purpose:
            - This function takes information from the company_npm_roa.csv and company_rev.csv and converts the data into a csv file named 'company_data_final.csv'
            - Data is taken from 2021-2022 in an annual measurement from Compustat North America Fundamental and Financial Ratio data bases
            - The file contained under file_path should contain:
                - GVKEY: company GVKEY number
                - CIK: company CIK number
                - NAICS: company North American Industry Classification System (NAICS) code
                - ($M) MCAP: estimated company market capitalization based on closed share price and total shares outstanding
                - ($M) REVT: company revenue total
                - ROA: company return on assets
                - NPM: company net profit margin
        """
        #obtain naics relations from 'company_info.csv'
        naics_d = dict()
        naics_set = set()
        for _, row in pd.read_csv(file_path).iterrows():
            comp = row['gvkey']
            if comp not in naics_set:
                naics_d[comp] = row['naics']
                naics_set.add(comp)

        #obtain ratio data
        ratio_df = pd.read_csv(file_path)
        ratio_df = ratio_df.dropna()
        ratio_set = set()
        ratio_dict = {} # this relates gvkey to npm and roa values in list
        for _, row in ratio_df.iterrows():
            comp = row['gvkey']
            if comp not in ratio_set:
                ratio_dict[comp] = (row['roa'], row['npm'])
                ratio_set.add(comp)

        #set up dictionary to be used to create pandas data frame
        d = {
            'GVKEY': [],
            'CIK': [],
            'NAICS': [],
            'MCAP': [],
            'REVT': [],
            'ROA': [],
            'NPM': [],
        }

        #obtain fundamentals data and create dictionary
        fund_df = pd.read_csv(file_path)
        fund_df = fund_df.dropna()
        fund_set = set()
        for _, row in fund_df.iterrows():
            comp = row['gvkey']
            if comp not in fund_set and comp in ratio_set:
                d['GVKEY'].append(comp)
                d['CIK'].append(row['cik'])
                d['NAICS'].append(naics_d[comp])
                d['REVT'].append(row['revt'])
                d['MCAP'].append(row['csho'] * row['prcc_f'])
                d['ROA'].append(ratio_dict[comp][0])
                d['NPM'].append(ratio_dict[comp][1])
                fund_set.add(comp)
        
        #create dictionary, drop rows with empty cells and save to csv file
        df = pd.DataFrame(d)
        df = df.dropna()
        df.to_csv('company_data_final.csv')

    def get_directors(self, file_path: str):
        """
        Parameters:
            - file_path: the directory to data repository
        Returns:
            - None
        Purpose:
            - Upload company financial performance data (GVKEY, CIK, NAICS, MCAP, REVT, ROA, NPM) alongside the directors' string
            - Will check directors against COMPUSTAT data for directors later
        """
        #initialize database connection
        excluded = 0
        director_count = 0
        
        #load variables from private .env file (can't be seen publicly on github)
        load_dotenv() 
        #create connection to PostgreSQL DB
        
        df_final = pd.read_csv(file_path)

        #go through each row
        for ix, row in df_final.iterrows():
            #Note the progress querying the api
            print("Progress:", 100*round(float(ix/len(df_final.index)), 3), "%")
            print(f"Searching for {row['GVKEY']}")

            #initialize constants
            directors = ''
            d_set = set()
            no_data = 0
            invalid_html = 0
            no_recent_filings = 0
            def14a_count = 0
            count = 0

            # Get the current date and time 3 years ago to this day
            now = datetime.now()
            two_years_ago = now - timedelta(days=365*2)

            #Initialize API call
            headers = {
                "User-Agent": "jmoore1@mit.edu"
            }
            zeroes = "0" * (10-len(str(int(row['CIK']))))
            url = f"https://data.sec.gov/submissions/CIK{zeroes}{int(row['CIK'])}.json"

            try:
                #get the response and index to recent filings in json
                response = r.get(url, headers=headers)
                info = response.json()['filings']['recent']
                
                for access_num, file, form, date in zip(info['accessionNumber'], info['primaryDocument'], info['form'], info['filingDate']):
                    #only allowed 10 requests/sec
                    time.sleep(0.1)

                    #need to stop search because we don't want to search forms older than 3 years old
                    if datetime.strptime(date, "%Y-%m-%d") < two_years_ago:
                        if not def14a_count:
                            print(f"No recent filings for {int(row['cik'])}")
                            no_recent_filings += 1
                        break
                    #don't check too many form 4's
                    elif count > 30:
                        break
                    #analyze the form
                    elif form == "4":
                        #initialize unique file url
                        f_url = f"https://www.sec.gov/Archives/edgar/data/{int(row['CIK'])}/{access_num.replace('-', '')}/{file}"
                        try:
                            f_response = r.get(f_url, headers=headers)
                            #Code to scrape directors names from FORM 4
                            soup = BeautifulSoup(f_response.text, 'html.parser')
                            name = soup.find_all("table")[4].find_all("a")[0].text
                            #find whether or not there's the 'X' next to the 'Director' option in the form 4
                            if soup.find_all("table")[4].find_all("table")[5].find_all("td")[0].text == 'X' and name not in d_set:
                                print(name, "Director added")
                                directors += name.lower() + ';'
                                d_set.add(name)
                            count += 1
                        except Exception as e:
                            #Can't read the html file
                            invalid_html = 1
                            print("Failed to load url! Exception: ", repr(e))
            except Exception as e:
                #couldn't access the link
                no_data = 1
                print(f"Couldn't retrive CIK data for {int(row['CIK'])}", repr(e))
            
            #create new DF, get rid of index column and export to db
            df = row.to_frame().T
            df.columns = df_final.columns
            #Add extra columns to DF
            df['Directors'] = [directors]
            df['No Data'] = [no_data]
            df['Invalid HTML'] = [invalid_html]
            df['No Recent Filings'] = [no_recent_filings]

            #export to PostgreSQL DB
            df.iloc[:, 1:].to_sql('company_data', self.engine, if_exists='append')
        
        #print summary
        print("Excluded", excluded, "companies")
        print("Found directors list for", director_count, "companies")

    def get_biographies(self, cik: str, directors: str):
        """
        Parameters:
            - cik: the company's unique CIK code
            - directors: a string of directors names separated by ';' character
        Returns:
            - director_bios: a dictionary that relates a director's name (FIRST LAST) to its respective biography in the DEF 14A filing
        Purpose:
            - For use in the calculation of tech savviness
        """

        # Retrieve the DEF 14A file from the EDGAR API
        # Initialize API call
        headers = {
            "User-Agent": "jmoore1@mit.edu"
        }
        zeroes = "0" * (10-len(str(int(cik))))
        url = f"https://data.sec.gov/submissions/CIK{zeroes}{int(cik)}.json"
        
        response = r.get(url, headers=headers)
        info = response.json()['filings']['recent']

        #clues and their respective weights
        clues = {'director sinc': 20, 'director of the company sinc': 20, 'elected a director': 20, 'appointed a director': 20, 'university': 10, 'college': 10, 'studi at': 10, 'director': 4, 'served a': 8, 'brings to the board': 10, 'founder of the compani': 10, 'founder': 5, 'qualif': 10, 'chairman of the board': 5, 'select': 1, 'nomin': 3, 'other board': 5, 'experi': 2, 'understand': 1, 'knowledg': 1, 'termin': -10, 'proxi': -10}
        
        #get directors from string (Ex. smith john h.;doe jane r.; ...)
        #Structure looks like [[last, first], [last, first], ...]
        directors_l = []
        for director in directors.split(';')[:-1]:
            #append first and last name strings into list
            directors_l.append(director.split(' ')[:2])

        for access_num, file, form, date in zip(info['accessionNumber'], info['primaryDocument'], info['form'], info['filingDate']):
            if form == 'DEF 14A':
                #create url and get a response
                f_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{access_num.replace('-', '')}/{file}"
                response = r.get(f_url, headers=headers)

                #make into html object and parse text using tokenizer
                soup = BeautifulSoup(response.text, 'html.parser')
                #make into lowercase and tokenize text
                parts = sent_tokenize(soup.text.lower())
                
                #initialize stemmer
                stemmer = PorterStemmer()
                
                #dictionary used to relate director to their respective highest-ranked biography
                director_bios = {}
                savvy = 0
                for director in directors_l:
                    #create list of sentences found in data structure: (sentence, rank)
                    sentence_record = []
                    #check every part in the file
                    for part in parts:
                        #need to check for every director, flag used because director name split into first and last
                        flag_director = False
                        for name in director:
                            if name in part:
                                flag_director = True
                        #only want to check parts which have directors names (first or last) in it
                        if flag_director:
                            rank = 0
                            stemmed = ' '.join([stemmer.stem(word) for word in nltk.word_tokenize(part)]) 
                            for clue in clues:
                                if clue in stemmed: 
                                    #add the weight to the total
                                    rank += clues[clue]
                            #append part with ranking and get rid of new line and non-breaking space characters
                            sentence_record.append((part.replace("\n", "").replace("\xa0", ""), rank))
                    
                    #sort list by second value in tuple
                    sentence_record.sort(key = lambda x: x[1], reverse=True)
                    #add director name (FIRST LAST) with sentence into dictionary
                    director_bios[f"{director[1]} {director[0]}"] = sentence_record[0][0]

                #return dictionary of names and related biographies for a single company
                return director_bios
        
    def find_savvy(self, fin_data: str, most_data: str, least_data: str):
        """
        Parameters:
            - fin_data: local filename for financial records for each company
            - most_data: local filename for the most indicative words for tech-savviness
            - least_data: local filename for the least indicative words for tech-savviness
        Returns:
            - None
        Purpose:
            - Parse director biographies and determine which directors are savvy or not based on keywords
            - Put data into PostgreSQL database for extraction and further analysis
        """
        
        df = pd.read_csv(fin_data)
        with open(most_data, 'r') as file:
            # Read all lines from the file into a list
            lines = file.readlines()
        # Remove the newline character ('\n') from each line and create a list
        most_sig = [line.rstrip('\n') for line in lines]

        with open(least_data, 'r') as file:
            # Read all lines from the file into a list
            lines = file.readlines()
        # Remove the newline character ('\n') from each line and create a list
        least_sig = [line.rstrip('\n') for line in lines]

        stemmer = PorterStemmer()

        for ix, row in df.iloc[722:, :].iterrows():
            try:
                bios = self.get_biographies(row['CIK'], row['Directors'])
                savvy = 0
                for bio in bios:
                    flag_savvy = False
                    for phrase in most_sig:
                        #create window with the same length as the phrase to search for the indicator phrases
                        if fuzz.partial_token_sort_ratio(' '.join([stemmer.stem(word) for word in phrase]), bio) > 66:
                            flag_savvy = True
                                    
                    if not flag_savvy:
                        count = 0
                        for phrase in least_sig:
                            if fuzz.partial_token_sort_ratio(' '.join([stemmer.stem(word) for word in phrase]), bio) > 66:
                                print(phrase)
                                if count >= 1:
                                    flag_savvy = True
                                    break
                                else:
                                    count += 1
                    savvy += flag_savvy
                    
                print("Progress:", ix / len(df.index), "Savvy:", savvy, "/", len(row['Directors'].split(';')[:-1]))
                # need to output to database
                df_row = pd.DataFrame([[row['CIK'], row['GVKEY'], row['NAICS'], row['MCAP'], row['REVT'], row['ROA'], row['NPM'], savvy, len(row['Directors'].split(';')[:-1])]], columns=['CIK', 'GVKEY', 'NAICS', 'MCAP', 'REVT', 'ROA', 'NPM', 'NUM_SAVVY', 'NUM_DIR'])
                df_row.to_sql('savvy_test_0', self.engine, if_exists='append', index=False)
            except:
                print("Error for", row['CIK'])
                continue

    def process_data(self, file_path: str):
        """
        Parameters:
            - file_path: the file path to the final data
            - fin_metric: the financial metric the Welch T-test is to be used on
        Purpose:
            - Filter each entry by a z-score of 3 (do this with a boolean mask in pandas)
            - Use stats.ttest_ind to determine significance of the two populations
        Returns:
            - JSON file with processed data
        """
        
        def create_final_rep(data_rep: dict, savvy: pd.DataFrame, not_savvy: pd.DataFrame):
            """
            Parameters:
                - data_rep: Dictionary that represents the JSON of the final calculated data
                - savvy: DataFrame that holds unprocessed data for savvy companies
                - not_savvy: DataFrame that holds unprocessed data for not savvy companies
            Purpose:
                - To store calcuated data into the data_rep for returning
            Returns:
                - data_rep which is the final JSON representation of processed data for output
            """

            def create_stats_simple(group_data: pd.DataFrame):
                """
                Parameters:
                    - a DataFrame which represents the financial data for all companies in a NAICS group
                Purpose:
                    - Given a NAICS groups stats compute the mean, median and St. Dev for each metric
                Returns:
                    - A dictionary which has all measurements from above given the data for the NAICS group
                """
                group_data = group_data.dropna()
                try:
                    return {'N': len(group_data.index),
                        'ROA mean': np.mean(group_data['ROA']),
                        'ROA median': np.median(group_data['ROA']),
                        'ROA standard deviation': np.std(group_data['ROA']),
                        'ROE mean': np.mean(group_data['ROE']),
                        'ROE median': np.median(group_data['ROE']),
                        'ROE standard deviation': np.std(group_data['ROE']),
                        'NPM mean': np.mean(group_data['NPM']),
                        'NPM median': np.median(group_data['NPM']),
                        'NPM standard deviation': np.std(group_data['NPM']),
                        'REVC mean': np.mean(group_data['REVCHANGE']),
                        'REVC median': np.median(group_data['REVCHANGE']),
                        'REVC standard deviation': np.std(group_data['REVCHANGE']),
                        'MCAP mean': np.mean(group_data['MCAP']),
                        'MCAP median': np.median(group_data['MCAP']),
                        'MCAP standard deviation': np.std(group_data['MCAP']),
                        'REVT mean': np.mean(group_data['REVT']),
                        'REVT median': np.median(group_data['REVT']),
                        'REVT standard deviation': np.std(group_data['REVT'])}
                except Exception as e:
                    print("Unable to create simple stats", e)
                    return None
                
            def create_stats_ttest(group_data: dict):
                """
                Parameters:
                    - group_data: dict which holds raw and stats about savvy and not savvy information
                Purpose:
                    - Need to obtain T-test statistics for every NAICS group
                Returns:
                    - Dictionary which holds information about T-test
                """
                try:
                    roa = stats.ttest_ind(group_data['savvy_raw']['ROA'], group_data['not_savvy_raw']['ROA'], equal_var = True)
                    roe = stats.ttest_ind(group_data['savvy_raw']['ROE'], group_data['not_savvy_raw']['ROE'], equal_var = True)
                    npm = stats.ttest_ind(group_data['savvy_raw']['NPM'], group_data['not_savvy_raw']['NPM'], equal_var = True)
                    mcap = stats.ttest_ind(group_data['savvy_raw']['MCAP'], group_data['not_savvy_raw']['MCAP'], equal_var = True)
                    revt = stats.ttest_ind(group_data['savvy_raw']['REVT'], group_data['not_savvy_raw']['REVT'], equal_var = True)
                    revc = stats.ttest_ind(group_data['savvy_raw']['REVCHANGE'], group_data['not_savvy_raw']['REVCHANGE'], equal_var = True)
                    return {'ROA t-statistic': roa[0], 'ROA p-value': roa[1],
                            'ROE t-statistic': roe[0], 'ROE p-value': roe[1],
                            'NPM t-statistic': npm[0], 'NPM p-value': npm[1],
                            'MCAP t-statistic': mcap[0], 'MCAP p-value': mcap[1],
                            'REVT t-statistic': revt[0], 'REVT p-value': revt[1],
                            'REVC t-statistic': revc[0], 'REVC p-value': revc[1],}
                except:
                    return None
                
            #CALCULATE THE OVERALL STATS
            data_rep['OVERALL'] = defaultdict()
            data_rep['OVERALL']['savvy'] = create_stats_simple(savvy.dropna())
            data_rep['OVERALL']['savvy_raw'] = savvy.dropna()
            data_rep['OVERALL']['not_savvy'] = create_stats_simple(not_savvy.dropna())
            data_rep['OVERALL']['not_savvy_raw'] = not_savvy.dropna()
            data_rep['OVERALL']['t_stats'] = create_stats_ttest(data_rep['OVERALL'])

            savvy = savvy.groupby(savvy['NAICS'].astype(str).str[:2])
            not_savvy = not_savvy.groupby(not_savvy['NAICS'].astype(str).str[:2])
            for group_name, group_data in savvy:
                data_rep[group_name] = defaultdict()
                data_rep[group_name]['savvy'] = create_stats_simple(group_data.dropna())
                data_rep[group_name]['savvy_raw'] = group_data.dropna()
            for group_name, group_data in not_savvy:
                if group_name not in data_rep:
                    data_rep[group_name] = defaultdict()
                data_rep[group_name]['not_savvy'] = create_stats_simple(group_data.dropna())
                data_rep[group_name]['not_savvy_raw'] = group_data.dropna()

            for naics_code, naics_data in data_rep.items():
                data_rep[naics_code]['t_stats'] = create_stats_ttest(naics_data)
                try: del data_rep[naics_code]['savvy_raw'] 
                except: continue
                try: del data_rep[naics_code]['not_savvy_raw']
                except: continue
            
            return data_rep

        df = pd.read_csv(file_path).dropna()
        data_rep = defaultdict()
        
        #filter data for savvy rows (companies only)
        savvy = df[df['NUM_SAVVY'] >= 3] 
        # create a boolean mask of rows which should be included
        z_scores = np.abs(stats.zscore(savvy)) < 3 
        mask = z_scores == False
        print("Number of Outliers for savvy", mask.sum().sum())
        #apply the mask
        savvy = savvy[z_scores]

        not_savvy = df[df['NUM_SAVVY'] < 3]
        z_scores = np.abs(stats.zscore(not_savvy)) < 3 
        mask = z_scores == False
        print("Number of Outliers for not_savvy", mask.sum().sum())
        not_savvy = not_savvy[z_scores]

        print("Total number of companies is", len(savvy.dropna().index) + len(not_savvy.dropna().index))

        data_rep = create_final_rep(data_rep, savvy, not_savvy)
        
        
        with open('result.json', 'w') as fp:
            json.dump(data_rep, fp, indent=2)


if __name__ == '__main__':
    None

    
    



    









    



    

    
    
    
        

    




