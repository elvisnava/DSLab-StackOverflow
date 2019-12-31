# DSLab-StackOverflow
Repository for our 2019 Data Science Lab project at ETH Zurich, titled "Moving towards Dynamic Recommendation in CQA"

# Installation

Install all requirements from the `requirements.txt`.   
Follow the tutorial `dump-to-postgres-tutorial.md` to convert the database dump from stackexchange to a postgresql database.   


# Baselines
### Choetkiertikul

Make sure that the postgresql server is running locally. 

Run `Baselines/choetkiertikul.py` to generate the file containing all user-question pairs used for training and testing (this takes a couple of hours). 
Run the notebook `data-exploration/choetkiertikul.ipynb` to train the random forest and generate plots. 


# Novel Approach
## GP-TOP-K
Run `gp_user2Lquestion.py` to run GP-TOP-K. A number of command line arguments are available. 


