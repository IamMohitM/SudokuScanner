import logging
import math

logger = logging.getLogger(__name__)


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"[{self.x}, {self.y}]"
    
    def __str__(self):
        return self.__repr__

    def is_left_of(self, point):
        return self.x > point.x

    def is_right_of(self, point):
        return self.x < point.x

    def is_above(self, point):
        return self.y < point.y
    
    def is_below(self, point):
        return self.y > point.y

    def is_horizontally_aligned(self, point):
        return self.x == point.x

    def is_vertically_aligned(self, point):
        return self.y == point.y

    def distance_with(self, point):
        return math.sqrt((self.x-point.x)**2 + (self.y-point.y)**2)


class Quadrilateral(object):
    def __init__(self, vertices) -> None:
        if len(vertices) != 4:
            raise ValueError(f"The vertices must be of length 4")

        self.vertices = vertices
        self.__compute_attributes__()


    def __compute_attributes__(self):
        sort_x = sorted(self.vertices, key = lambda x: x[0])

        if (l_1 := sort_x[0])[1] < (l_2 := sort_x[1])[1]:
            top_left, bottom_left = l_1, l_2
        else:
            top_left, bottom_left = l_2, l_1

        if (r_1 := sort_x[2])[1] < (r_2 := sort_x[3])[1]:
            top_right, bottom_right = r_1, r_2
        else:
            top_right, bottom_right = r_2, r_1

        self.top_left = top_left
        self.top_right = top_right
        self.bottom_left = bottom_left
        self.bottom_right = bottom_right

        self.ordered_vertices = [self.top_left, self.top_right, self.bottom_right, self.bottom_left]

        self.lengths = [self.distance(p, q) for p,q in zip(self.ordered_vertices, 
            self.ordered_vertices[1:] + [self.top_left])]


    def __str__(self):

        try:
             return self.__pr_string__
        
        except AttributeError:
            pr_string = "["

            for v in self.vertices[:-1]:
                pr_string += f"{v}, "
            pr_string += f"{self.vertices[-1]}]"

            self.__pr_string__ = pr_string
            return self.__pr_string__

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self):
        return 4

    def distance(self, p, q):
        return math.sqrt((p[0]-q[0])**2 + (q[1]-p[1])**2)

    def find_inside(self, quad):
        pass
        
    def intersection_area(self, quad):
        
        raise NotImplementedError()
        pass
    


    