from .render import render
from .neilf import render_neilf


render_fn_dict = {
    "render": render,
    "neilf": render_neilf,
}