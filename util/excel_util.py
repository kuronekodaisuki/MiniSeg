import os
from os import path
import openpyxl as op
import pandas as pd

def generate_excel(input_dict, out_path, total_blocks):
    data = dict()
    names = list()
    steps = list()
    times = list()
    for key in input_dict.keys():
        item_d = input_dict[key]
        for sub_key in item_d.keys():
            names.append(key)
            steps.append(sub_key)
            times.append(item_d[sub_key])
    data['name'] = names
    data['step'] = steps
    data['runtime(ms)'] = times
    df = pd.DataFrame(data)
    body_number = names[0].split('_')[1]
    out_file = path.join(out_path, f"{body_number}_{total_blocks}_runtime_statics.xlsx")
    df.to_excel(out_file, index=False)
