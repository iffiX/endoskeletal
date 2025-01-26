from typing import List, Dict, Any


def list_of_dict_to_dict_of_list(list_of_dicts: List[Dict[str, Any]]):
    keys = list(list_of_dicts[0].keys())
    new_dict = {}
    for key in keys:
        new_dict[key] = []
        for sub_dict in list_of_dicts:
            new_dict[key].append(sub_dict[key])
    return new_dict
