import csv
from pathlib import Path
import sys
import os
import django

# Configuracion con el entorno de Django
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'idealista.settings')
django.setup()

from webapp.models import Vivienda

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

            for fila in lector:
                vivienda = Vivienda(
                    id=fila["ASSETID"],
                    precio=int(fila["PRICE"]),
                    precio_m2=float(fila["UNITPRICE"]),
                    metros_construidos=int(fila["CONSTRUCTEDAREA"]),
                    num_hab=int(fila["ROOMNUMBER"]),
                    num_wc=int(fila["BATHNUMBER"]),
                    terraza=bool((fila["HASTERRACE"])),
                    ascensor=bool((fila["HASLIFT"])),
                    aire_acondicionado=bool((fila["HASAIRCONDITIONING"])),
                    parking=bool((fila["HASPARKINGSPACE"])),
                    orientacion_norte=bool((fila["HASNORTHORIENTATION"])),
                    orientacion_sur=bool((fila["HASSOUTHORIENTATION"])),
                    orientacion_este=bool((fila["HASEASTORIENTATION"])),
                    orientacion_oeste=bool((fila["HASWESTORIENTATION"])),
                    trastero=bool((fila["HASBOXROOM"])),
                    armario_empotrado=bool((fila["HASWARDROBE"])),
                    piscina=bool((fila["HASSWIMMINGPOOL"])),
                    portero=bool((fila["HASDOORMAN"])),
                    jardin=bool((fila["HASGARDEN"])),
                    duplex=bool((fila["ISDUPLEX"])),
                    estudio=bool((fila["ISSTUDIO"])),
                    ultima_planta=bool((fila["ISINTOPFLOOR"])),
                    planta=int(fila["FLOORCLEAN"]),
                    anyo_catastro=int(fila["CADCONSTRUCTIONYEAR"]),
                    plantas_edicio_catastro=int(fila["CADMAXBUILDINGFLOOR"]),
                    viviendas_edificio_catastro=int(fila["CADDWELLINGCOUNT"]),
                    calidad_catastro=int(fila["CADASTRALQUALITYID"]),
                    distancia_centro=float(fila["DISTANCE_TO_CITY_CENTER"]),
                    distancia_metro=float(fila["DISTANCE_TO_METRO"]),
                    distancia_blasco=float(fila["DISTANCE_TO_BLASCO"]),
                    longitud=float(fila["LONGITUDE"]),
                    latitud=float(fila["LATITUDE"]),
                )
                vivienda.save()
                viviendas_creadas += 1

            print(f"Se han cargado {viviendas_creadas} viviendas en la base de datos.")
    except FileNotFoundError:
        print(f"El archivo {ruta_csv} no fue encontrado.")
    except Exception as e:
        print(f"Ocurrió un error al cargar los datos: {e}")

if __name__ == "__main__":
    ruta_csv = BASE_DIR / "data_pipeline" / "processed_data" / "Valencia_Sale_it1.csv"
    cargar_viviendas_desde_csv(ruta_csv)