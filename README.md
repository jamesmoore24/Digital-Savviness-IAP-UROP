# Natural Language Processing of Board of Director Biographies

Digital savviness among board of directors has been linked to financial success of several publicy-listed companies per a study done in 2017 led by the MIT Center for Information Systems Research. Due to recent advances in technology, it has become clear that the indicators of potentially tech-savvy directors has changed which has prompted the creation of a replicate study which will be documented here. 

## About:

Uses popular python data processing packages such as ```pandas```, ```numpy``` and ```mathplotlib``` through the ```scipy``` distribution to carry out data cleaning, analysis and visualization. Uses [COMPUSTAT](https://www.spglobal.com/marketintelligence/en/?product=compustat-research-insight) to obtain financial performance information of publicly-traded companies. Uses [```NLTK```](https://www.nltk.org/) and SeetGeek's [```fuzzywuzzy```](https://github.com/seatgeek/fuzzywuzzy) package to parse biography and assess individual's tech savviness. Also uses the new [EDGAR API](https://www.sec.gov/edgar/sec-api-documentation) to search company submitted SEC files which will be used to identify director names from form 4 filings and schedule 14A proxies which contain board of director biographies

---

## Documentation

### Getting Started:

- This program can run on Windows, Mac and Linux operating system by using valid python text editors with the python version 3.x
- This program is developed by python 3.x.
- Must download ```pandas```, ```mathplotlib```, ```numpy```, ```stanza``` using [pip](https://pip.pypa.io/en/stable/) if not using [Anaconda Environment](https://docs.continuum.io/anaconda/), alternative ways to obtain packages are [here](https://scipy.org/install/).

### Problem Statement: 

Need to determine whether digital-savviness among a board of directors at a given company is significantly correlated to the company's financial success.

### Approach:

COMPLETED:
1. Connect to, investigate, and test API data vendors that they are sending complete data.
2. Complete review of existing R files used for data collection and processing and update techniques using Python.
3. Implement process for evaluating tech-savviness:
    * Pull biographies from SEC's EDGAR database
    * Utilize TLNK's NLP library to parse biographies
4. Update indicator words/phrases for use in digital-savviness evaluation.
5. Implement process for evaluating tech-savviness:
    * Decide on thresholds for NLP evaluation and assign flag (tech-savvy or not tech-savvy) to each company being investigated
    * Map NLP evaluation to financial performance indicators 
    * Use T-test to determine if significant financial performance difference exists between the two groups.
6. Generate data visuals (word cloud, histogram, box plots, etc...) to demonstrate findings
7. Organize and detail quantitative and visual results in academic briefing in LATEX


### Takeaways:
This project taught me more about how to format and utilize different data structures to store and transport information from APIs to the resulting product. I also gained new experience with using the [```NLTK```](https://www.nltk.org/) library and implementing different types of searches using [```fuzzywuzzy```](https://github.com/seatgeek/fuzzywuzzy) package. Data processing and handling using ```pandas``` and statistical caluclations using ```numpy``` proved to be self-explanatory and helpful in reduce code complexity and increasing readability which resulted in a 91.803% decrease in lines of code (5636 lines to 462) from R legacy codebase.
