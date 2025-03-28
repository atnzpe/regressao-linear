import flet as ft

colors = {
    "azul": "#3155A4",
    "amarelo": "#FFB511",
    "vermelho": "#C34342",
    "verde": "#00AD4A",
    "branco": "#FFFFFF",
}

dimensions = {
    "base_height": 900,
    "base_width": 420,
    "button_width": 350,
    "button_height": 50,
    "border_radius": 30,
    "content_padding": ft.padding.only(left=20, top=10, right=10, bottom=10),
}

def get_control_name(self):
    return ""