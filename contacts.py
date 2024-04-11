"""
Generates a list of propositions from a loaded csv file
"""
from pathlib import Path
import pandas as pd
class CSV:
    def __init__(self, file_path:Path):
        self.data = pd.read_csv(str(file_path), encoding='utf-8')
        names = self.data.Name
        self.contacts = set()
        for name in names:        
            if isinstance(name, str): # reject nan
                self.contacts.add(name)    

if __name__ == "__main__":
    csv = CSV("contacts (4).csv")
    print(csv.contacts)