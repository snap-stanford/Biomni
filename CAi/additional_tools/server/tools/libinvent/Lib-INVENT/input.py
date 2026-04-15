#!/usr/bin/env python

import sys
import json
from running_modes.manager import Manager


if __name__ == "__main__":

    with open(sys.argv[1]) as f:
        json_input = f.read().replace('\r', '').replace('\n', '')

    configuration = {}
    try:
        configuration = json.loads(json_input)
    except (ValueError, KeyError, TypeError):
        print("JSON format error")
    else:
        manager = Manager(configuration)
        manager.run()
