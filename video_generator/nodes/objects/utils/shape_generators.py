from turtle import Turtle
from functools import wraps

from typing import List

def apply_styles(
    turtle: Turtle, 
    fill_color: str = 'white', 
    border_color: str = 'black', 
    border_width: int = 1,
    **kwargs
):
    turtle.fillcolor(fill_color)
    turtle.pencolor(border_color)
    turtle.pensize(border_width) 

def render_fn(fn): 
    @wraps(fn)
    def wrapped_fn(turtle: Turtle, start: tuple, *args, **kwargs):
        turtle.penup()
        turtle.goto(start)
        apply_styles(turtle, **kwargs)
        turtle.pendown()
        turtle.begin_fill()
        fn(turtle, start, *args, **kwargs)
        turtle.end_fill()
        turtle.penup()

    return wrapped_fn

@render_fn
def draw_circle(turtle: Turtle, start: tuple, radius: int, **kwargs):
    turtle.circle(radius = radius)

@render_fn
def draw_polygon(turtle: Turtle, start: tuple, other_points: List[tuple], **kwargs): 
    for point in other_points: 
        turtle.goto(point)
    turtle.goto(start)
    
@render_fn
def draw_line(turtle: Turtle, start: tuple, end: tuple, **kwargs):
    turtle.goto(end)


def draw_rectangle(turtle: Turtle, start: tuple, height: int, width: int, **kwargs): 
    x,y = turtle.xcor, turtle.ycor
    rectangle_pnts = [
        (x + width, y), 
        (x + width, y - height),
        (x, y - height)
    ]
    draw_polygon(turtle, (x,y), rectangle_pnts, **kwargs)

