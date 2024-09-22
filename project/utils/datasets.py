import os

def build_dataHomeFolder():
    if not os.path.exists('././datasets'):
        os.makedirs('././datasets')
    return '././datasets/'

