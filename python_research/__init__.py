from enum import Enum

# Magenta / Fuchsia	#FF00FF	(255,0,255)
#  	Silver	#C0C0C0	(192,192,192)
#  	Gray	#808080	(128,128,128)
#  	Maroon	#800000	(128,0,0)
#  	Olive	#808000	(128,128,0)
#  	Green	#008000	(0,128,0)
#  	Purple	#800080	(128,0,128)
#  	Teal	#008080	(0,128,128)
#  	Navy	#000080	(0,0,128)
class Color(Enum):

    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    CYAN = (255, 255, 0)
    MAROON = (0, 0, 128)
    OLIVE = (0, 128, 128)
    PURPLE = (128, 0, 128)
    TEAL = (128, 128, 0)
    NAVY = (128, 0, 0)
    MAGENTA = (255, 0, 255)
    YELLOW = (0, 255, 255)


