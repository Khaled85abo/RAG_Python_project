import os
import platform

def clear_screen():
    system = platform.system()
    if system == 'Windows':
        os.system("cls")
    else :
        os.system("clear")
