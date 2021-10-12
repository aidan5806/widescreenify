from utils import shape_generators

def to_node_cfg(generator_fn, start, **kwargs): 
    return dict(
        generator_fn = generator_fn, 
        start = start, 
        **kwargs
    )

def circle_cfg(start, radius, **kwargs):
    kwargs['radius'] = radius
    return to_node_cfg(shape_generators.draw_circle, start, **kwargs)

