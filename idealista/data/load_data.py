import csv
from pathlib import Path
import sys
import os
import django
# Configuracion del entorno Django
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'idealista.settings')
django.setup()

from webapp.models import Barriada, Vivienda, Metro, Hiperparametro

def cargar_barrios_desde_csv(ruta_csv):
    """
    Cargar los datos de un archivo CSV de barrios en la base de datos
    de Django, transformando los datos al formato del modelo creado.

    Parametros:
    ruta_csv (str): Ruta al archivo CSV que contiene los datos de los barrios.

    Returns:
    Ninguno
    """
    with open(ruta_csv, mode='r', encoding='utf-8') as archivo:
        lector = csv.DictReader(archivo)
        barrios = 0
        for fila in lector:
            barrio = Barriada(
                id=fila["NEIGHBOURID"],
                nombre=fila["NEIGHBOURNAME"]
            )
            barrio.save()
            barrios += 1

        print(f"Se han creado {barrios} barrios nuevos.")

def cargar_hiperparametros_desde_csv(ruta_csv):
    """
    Carga los hiperparámetros de un archivo CSV y los guarda en la base de datos,
    asociando cada conjunto de hiperparámetros al barrio correspondiente.

    Parametros:
    ruta_csv (str): Ruta al archivo CSV con los campos: ID,hidden_channels,num_layers,dropout,lr,epochs

    Returns:
    Ninguno
    """
    with open(ruta_csv, mode='r', encoding='utf-8') as archivo:
        lector = csv.DictReader(archivo)
        hiperparametros_creados = 0
        for fila in lector:
            try:
                barrio = Barriada.objects.get(id=fila["ID"])
                hiper = Hiperparametro(
                    barrio=barrio,
                    hidden_channels=int(fila["HIDDENCHANELS"]),
                    num_layers=int(fila["LAYERS"]),
                    dropout=float(fila["DROPOUT"]),
                    lr=float(fila["LR"]),
                    epochs=int(fila["EPOCHS"])
                )
                hiper.save()
                hiperparametros_creados += 1
            except Barriada.DoesNotExist:
                print(f"Barrio con ID {fila['ID']} no encontrado. Saltando fila.")
            except Exception as e:
                print(f"Error al guardar hiperparámetro para barrio {fila['ID']}: {e}")
        print(f"Se han cargado {hiperparametros_creados} hiperparámetros en la base de datos.")

def cargar_viviendas_desde_csv(ruta_csv):
    """
    Carga los datos de un archivo CSV preprocesado y preparado para
    ser importado a la base de datos.

    Parametros:
    ruta_csv (str): Ruta al archivo CSV que contiene los datos de las viviendas.

    Returns:
    Ninguno
    """
    try:
        with open(ruta_csv, mode='r', encoding='utf-8') as archivo:
            lector = csv.DictReader(archivo)
            viviendas_creadas = 0

            def bool_from_csv(val):
                return str(val).strip() == "1"

            for fila in lector:
                # Determina atributo 'estado'
                if fila.get("BUILTTYPEID_1") == "1":
                    status_value = "NEWCONSTRUCTION"
                elif fila.get("BUILTTYPEID_2") == "1":
                    status_value = "2HANDRESTORE"
                else:
                    status_value = "2HANDGOOD"

                vivienda = Vivienda(
                    id=fila["ASSETID"],
                    precio_m2=float(fila["UNITPRICE"]),
                    metros_construidos=int(fila["CONSTRUCTEDAREA"]),
                    num_hab=int(fila["ROOMNUMBER"]),
                    num_wc=int(fila["BATHNUMBER"]),
                    terraza=bool_from_csv(fila["HASTERRACE"]),
                    ascensor=bool_from_csv(fila["HASLIFT"]),
                    aire_acondicionado=bool_from_csv(fila["HASAIRCONDITIONING"]),
                    parking=bool_from_csv(fila["HASPARKINGSPACE"]),
                    orientacion_norte=bool_from_csv(fila["HASNORTHORIENTATION"]),
                    orientacion_sur=bool_from_csv(fila["HASSOUTHORIENTATION"]),
                    orientacion_este=bool_from_csv(fila["HASEASTORIENTATION"]),
                    orientacion_oeste=bool_from_csv(fila["HASWESTORIENTATION"]),
                    trastero=bool_from_csv(fila["HASBOXROOM"]),
                    armario_empotrado=bool_from_csv(fila["HASWARDROBE"]),
                    piscina=bool_from_csv(fila["HASSWIMMINGPOOL"]),
                    portero=bool_from_csv(fila["HASDOORMAN"]),
                    jardin=bool_from_csv(fila["HASGARDEN"]),
                    duplex=bool_from_csv(fila["ISDUPLEX"]),
                    estudio=bool_from_csv(fila["ISSTUDIO"]),
                    ultima_planta=bool_from_csv(fila["ISINTOPFLOOR"]),
                    planta=int(fila["FLOORCLEAN"]),
                    plantas_edicio_catastro=int(fila["CADMAXBUILDINGFLOOR"]),
                    calidad_catastro=int(fila["CADASTRALQUALITYID"]),
                    distancia_centro=float(fila["DISTANCE_TO_CITY_CENTER"]),
                    distancia_metro=float(fila["DISTANCE_TO_METRO"]),
                    distancia_blasco=float(fila["DISTANCE_TO_BLASCO"]),
                    longitud=float(fila["LONGITUDE"]),
                    latitud=float(fila["LATITUDE"]),
                    estado=status_value,
                    antiguedad=int(fila["ANTIQUITY"]),
                    barrio=Barriada.objects.get(id=fila["NEIGHBOURID"]),
                )
                vivienda.save()
                viviendas_creadas += 1
            print(f"Se han cargado {viviendas_creadas} viviendas en la base de datos.")
    except FileNotFoundError:
        print(f"El archivo {ruta_csv} no fue encontrado.")
    except Exception as e:
        print(f"Ocurrio un error al cargar los datos: {e}")

def cargar_metros_desde_csv(ruta_csv):
    """
    Carga los datos de bocas de metro desde un archivo CSV y los guarda en la base de datos.

    El CSV debe tener las columnas: NAME, LATITUDE, LONGITUDE (cabeceras).
    """
    import csv

    try:
        with open(ruta_csv, mode='r', encoding='utf-8') as archivo:
            lector = csv.DictReader(archivo)
            metros_creados = 0
            for fila in lector:
                nombre = fila.get("NAME") or fila.get("Denominació / Denominación")
                latitud = fila.get("LATITUDE")
                longitud = fila.get("LONGITUDE")
                if not (nombre and latitud and longitud):
                    continue
                metro = Metro(
                    nombre=nombre,
                    latitud=float(latitud),
                    longitud=float(longitud)
                )
                metro.save()
                metros_creados += 1
            print(f"Se han cargado {metros_creados} bocas de metro en la base de datos.")
    except FileNotFoundError:
        print(f"El archivo {ruta_csv} no fue encontrado.")
    except Exception as e:
        print(f"Ocurrió un error al cargar los datos de metro: {e}")

if __name__ == "__main__":
    datos_barrios = BASE_DIR / "data" / "db" / "barris-barrios_data.csv"
    datos_viviendas = BASE_DIR / "data" /  "db" / "Valencia_Sale_data.csv"
    datos_hiperparametros = BASE_DIR / "data" / "db" / "hiperparametros.csv"
    datos_metro = BASE_DIR / "data" / "db" / "bocas_metro.csv"
    cargar_barrios_desde_csv(datos_barrios)
    cargar_hiperparametros_desde_csv(datos_hiperparametros)
    cargar_viviendas_desde_csv(datos_viviendas)
    cargar_metros_desde_csv(datos_metro)
    print("Carga de datos completada.")