import flet as ft
import pandas as pd
import matplotlib.pyplot as plt
import io
from PIL import Image
import numpy as np
from linear_regression import LinearRegression
from logging import error, info, basicConfig, DEBUG, FileHandler, StreamHandler
import extras

# Configuração do logging
basicConfig(
    level=DEBUG,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s",
    handlers=[
        FileHandler("app.log"),
        StreamHandler(),
    ],
)


def executar_regressao(
    page: ft.Page,
    texto_etapas: ft.Text,
    carregando: ft.Ref[ft.ProgressRing],
    progresso: ft.ProgressBar,
    texto_x_novo: ft.TextField,
    file_path: str,
):
    """Executa a regressão linear."""
    info("Iniciando execução da regressão.")
    if file_path is None:
        texto_etapas.value = "Selecione um arquivo antes de calcular a regressão."
        page.update()
        info("Nenhum arquivo selecionado.")
        return

    carregando.current.visible = True
    page.update()
    info("ProgressRing visível.")
    try:
        info(f"Lendo dados do arquivo: {file_path}")
        x, y = ler_dados_do_arquivo(file_path)
        info("Dados lidos com sucesso.")
        progresso.value = 0.2
        page.update()
        info("Barra de progresso atualizada (20%).")
        model = LinearRegression(np.array(x), np.array(y))
        info("Modelo de regressão criado.")
        progresso.value = 0.5
        page.update()
        info("Barra de progresso atualizada (50%).")
        try:
            x_novo = float(texto_x_novo.value)
            info(f"Valor de X fornecido: {x_novo}")
        except ValueError:
            texto_etapas.value = "Valor de X inválido. Digite um número."
            page.update()
            info("Valor de X inválido.")
            return

        progresso.value = 0.8
        page.update()
        info("Barra de progresso atualizada (80%).")
        previsao = model.predict(x_novo)
        info(f"Previsão calculada: {previsao}")
        progresso.value = 1.0
        page.update()
        info("Barra de progresso atualizada (100%).")

        try:
            info("Gerando gráfico...")
            plt.figure(figsize=(6, 4))
            plt.scatter(x, y, label="Dados", color=extras.colors["amarelo"])
            x_line = np.linspace(min(x), max(x), 100)
            y_line = model.predict(x_line)
            plt.plot(
                x_line, y_line, color=extras.colors["verde"], label="Regressão Linear"
            )
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend()
            plt.title("Regressão Linear")

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            image = Image.open(buf)
            image_bytes = image.tobytes("raw")

            resultados = ft.Column(
                [
                    ft.Text(
                        f"Valores de X: {list(model.x)}", color=extras.colors["amarelo"]
                    ),
                    ft.Text(
                        f"Valores de Y: {list(model.y)}", color=extras.colors["amarelo"]
                    ),
                    ft.Text(
                        f"Coeficiente de Correlação: {model._correlation_coefficient:.4f}",
                        color=extras.colors["amarelo"],
                    ),
                    ft.Text(
                        f"Inclinação: {model._inclination:.4f}",
                        color=extras.colors["amarelo"],
                    ),
                    ft.Text(
                        f"Intercepto: {model._intercept:.4f}",
                        color=extras.colors["amarelo"],
                    ),
                    ft.Text(
                        f"Previsão para x = {x_novo}: {previsao:.4f}",
                        color=extras.colors["amarelo"],
                    ),
                ],
                spacing=10,
            )

            page.controls.append(resultados)
            page.controls.append(
                ft.Image(image_bytes, width=300, height=200, fit=ft.ImageFit.CONTAIN)
            )
            page.update()
            info("Resultados exibidos com sucesso.")

        except Exception as e:
            texto_etapas.value = f"Erro ao gerar o gráfico: {e}"
            error(f"Erro ao gerar gráfico: {e}")
            page.update()
            return

    except FileNotFoundError:
        texto_etapas.value = "Arquivo não encontrado. Verifique o caminho."
        page.update()
        info("Arquivo não encontrado.")
    except pd.errors.EmptyDataError:
        texto_etapas.value = "O arquivo está vazio."
        page.update()
        info("Arquivo vazio.")
    except pd.errors.ParserError:
        texto_etapas.value = "Erro ao analisar o arquivo. Verifique o formato."
        page.update()
        info("Erro de análise do arquivo.")
    except KeyError as e:
        texto_etapas.value = f"Coluna '{e.args[0]}' não encontrada no arquivo."
        page.update()
        info(f"Coluna '{e.args[0]}' não encontrada.")
    except Exception as e:
        error(f"Erro durante a execução da regressão: {e}")
        texto_etapas.value = f"Erro inesperado: {e}"
        page.update()
        info(f"Erro inesperado: {e}")
    finally:
        carregando.current.visible = False
        page.update()
        info("ProgressRing escondido.")


def ler_dados_do_arquivo(file_path):
    try:
        df = pd.read_csv(file_path, sep=";", header=None, names=["x", "y"])
        x = df["x"].tolist()
        y = df["y"].tolist()
        return x, y
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo '{file_path}' não encontrado.")
    except pd.errors.EmptyDataError:
        raise ValueError("O arquivo está vazio.")
    except pd.errors.ParserError:
        raise ValueError(
            "Erro ao analisar o arquivo. Verifique se ele está no formato correto (duas colunas separadas por ';')."
        )
    except KeyError as e:
        raise KeyError(f"Coluna '{e.args[0]}' não encontrada no arquivo.")
    except Exception as e:
        raise Exception(f"Erro imprevisto ao ler o arquivo: {e}")


def main(page: ft.Page):
    page.title = "Cálculo Regressão Linear"
    page.bgcolor = extras.colors["azul"]
    page.window_width = extras.dimensions["base_width"]
    page.window_height = extras.dimensions["base_height"]
    page.window_resizable = False

    texto_etapas = ft.Text(
        value="Aguardando informações...",
        size=16,
        color=extras.colors["amarelo"],
    )
    carregando = ft.Ref[ft.ProgressRing]()
    file_picker = ft.FilePicker(
        on_result=lambda e: setattr(
            page, "file_path", e.files[0].path if e.files else None
        )
    )
    texto_x_novo = ft.TextField(
        label="Variável Independente:",
        width=100,
        border_radius=extras.dimensions["border_radius"],
        content_padding=extras.dimensions["content_padding"],
        fill_color=extras.colors["branco"],
    )
    texto_x_novo.content = ft.Text(value="", color=extras.colors["azul"])

    botao_selecionar_arquivo = ft.ElevatedButton(
        "Selecionar Arquivo", on_click=lambda _: file_picker.pick_files()
    )
    botao_calcular = ft.ElevatedButton(
        "Calcular Regressão",
        width=extras.dimensions["button_width"],
        height=extras.dimensions["button_height"],
        bgcolor=extras.colors["verde"],
        color=extras.colors["branco"],
        style=ft.ButtonStyle(
            shape={
                "": ft.RoundedRectangleBorder(radius=extras.dimensions["border_radius"])
            }
        ),
        on_click=lambda _: executar_regressao(
            page,
            texto_etapas,
            carregando,
            progresso,
            texto_x_novo,
            getattr(page, "file_path", None),
        ),
    )

    coluna_botoes = ft.Column(
        [botao_selecionar_arquivo, botao_calcular],
        alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
    )

    progresso = ft.ProgressBar(width=extras.dimensions["button_width"])
    carregando_ref = ft.ProgressRing(ref=carregando)

    def fechar_alerta(e):
        page.dialog.open = False
        page.dialog = None
        page.update()

    alerta = ft.AlertDialog(
        modal=True,
        title=ft.Text(
            "Instruções",
            size=20,
            weight="bold",
            text_align=ft.TextAlign.CENTER,
            color=extras.colors["amarelo"],
        ),
        content=ft.Column(
            [
                ft.Text(
                    "O arquivo deve conter duas colunas (x e y) separadas por ponto e vírgula (;).",
                    size=15,
                    weight="italic",
                    text_align=ft.TextAlign.JUSTIFY,
                    color=extras.colors["branco"],
                ),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        actions=[
            ft.ElevatedButton(
                "Fechar",
                on_click=fechar_alerta,
                color=extras.colors["branco"],
            )
        ],
    )

    page.dialog = alerta
    alerta.open = True
    page.update()

    page.add(
        ft.Container(
            ft.Column(
                [
                    coluna_botoes,
                    texto_x_novo,
                    texto_etapas,
                    progresso,
                    file_picker,
                    carregando_ref,
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=extras.dimensions["content_padding"],
            width=extras.dimensions["base_width"],
            height=extras.dimensions["base_height"],
            border_radius=extras.dimensions["border_radius"],
        )
    )


ft.app(target=main)
