from pathlib import Path
import pandas as pd
import csv
import geopandas as gpd
import shapely.geometry
import json
from sklearn.preprocessing import MinMaxScaler
import os

def procesar_csv_viviendas(ruta_entrada, ruta_salida):
    """
    Unifica la funcion de lectura y reescritura del archivo CSV en sucio.

    La funcion procesa un archivo CSV de entrada, seleccionando todas las columnas
    a excepcion de "geometry", la cual da problemas al leer el archivo con pandas.
    Luego, escribe el contenido en un nuevo archivo CSV de salida.

    Parametros:
    ruta_entrada (str): Ruta al archivo CSV de entrada.
    ruta_salida (str): Ruta al archivo CSV de salida.

    Returns:
    Ninguno
    """
    def leer_csv(ruta_archivo):
        """
        Funcion interna para leer un archivo CSV en sucio y eliminar la columna "geometry".
        Convierte el CSV en una lista de diccionarios, donde cada diccionario representa una fila del CSV.

        Parametros:
        ruta_archivo (str): Ruta al archivo CSV.

        Returns:
        list: Lista de diccionarios con los datos del archivo CSV.
        """
        datos = []
        try:
            with open(ruta_archivo, mode='r', encoding='utf-8') as archivo:
                lector = csv.reader(archivo)
                encabezados = next(lector) # La primera fila es la de encabezados
                encabezados = encabezados[:41] # Elimina la columna "geometry"
                for fila in lector:
                    vivienda = {encabezados[i]: fila[i] for i in range(len(encabezados))}
                    datos.append(vivienda)
        except FileNotFoundError:
            print(f"El archivo {ruta_archivo} no fue encontrado.") 
        except Exception as e:
            print(f"Ocurrio un error al leer el archivo: {e}")
        return datos

    def escribir_csv(ruta_salida, datos, encabezados):
        """
        Funcion interna para escribir un archivo CSV a partir de una lista de diccionarios.
        Convierte la lista de diccionarios en un archivo CSV, donde cada diccionario representa una fila.
        Devolverá un mensaje de exito o error al escribir el archivo.

        Parametros:
        ruta_salida (str): Ruta al archivo CSV de salida, donde se guardara el archivo.
        datos (list): Lista de diccionarios con los datos a escribir.
        encabezados (list): Lista de encabezados para el archivo CSV.

        Returns:
        Ninguno
        """
        try:
            with open(ruta_salida, mode='w', encoding='utf-8', newline='') as archivo:
                escritor = csv.DictWriter(archivo, fieldnames=encabezados)
                escritor.writeheader()  # Encabezados
                escritor.writerows(datos)  # Filas
            print(f"Archivo CSV creado correctamente en /processed_data")
        except Exception as e:
            print(f"Ocurrio un error al escribir el archivo: {e}")

    # Ruta de salida
    Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)

    contenido = leer_csv(ruta_entrada)

    if contenido:
        encabezados = list(contenido[0].keys())
        escribir_csv(ruta_salida, contenido, encabezados)

def procesar_barrios_csv(ruta_csv, ruta_csv_salida, ruta_geojson_salida):
    """
    Lee un CSV de barrios y realiza el preprocesamiento necesario para convertirlo
    en un GeoDataFrame. Elimina columnas innecesarias, renombra columnas y convierte
    la geometría de GeoJSON a objetos shapely.

    Guarda el resultado en un nuevo CSV y un archivo GeoJSON.

    Parametros:
    ruta_csv (str): Ruta al archivo CSV de barrios.
    ruta_csv_salida (str): Ruta al archivo CSV de salida.
    ruta_geojson_salida (str): Ruta al archivo GeoJSON de salida.

    Returns:
    Ninguno
    """
    df = pd.read_csv(ruta_csv, sep=';', dtype=str)

    # Columnas que no se necesitan
    df = df.drop(columns=['codbarrio', 'gis_gis_barrios_area', 'geo_point_2d', 'coddistrit'])

    # Se renombran las columnas para que siga el formato del CSV de viviendas
    df = df.rename(columns={
        'coddistbar': 'NEIGHBOURID',
        'nombre': 'NEIGHBOURNAME',
        'geo_shape': 'geometry'
    })

    # Identificadores únicos para cada barrio
    df['NEIGHBOURID'] = range(1, len(df) + 1)

    df[['NEIGHBOURID', 'NEIGHBOURNAME']].to_csv(ruta_csv_salida, index=False, encoding='utf-8')

    # Se convierte la columna geometry (GeoJSON string) a objetos shapely
    def geojson_to_shape(geojson_str):
        try:
            geojson = json.loads(geojson_str)
            return shapely.geometry.shape(geojson)
        except Exception:
            return None

    df['geometry'] = df['geometry'].apply(geojson_to_shape)

    # Se crea el GeoJSON
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    gdf.to_file(ruta_geojson_salida, driver='GeoJSON')

def preprocesamiento(ruta_viviendas, ruta_salida_viviendas, ruta_barrios, ruta_salida_no_normalizado):
    """
    Preprocesa el archivo CSV de entrada, eliminando atributos considerados
    como innecesarios para el contexto del problema, ademas de eliminar filas
    donde los valores sean nulos, vacios o incongruentes. Convierte ciertos
    atributos a tipos especificos.

    La funcion tambien normaliza ciertos atributos y guarda el resultado en un nuevo archivo CSV.

    Parametros:
    ruta_archivo (str): Ruta al archivo CSV.
    ruta_salida (str): Ruta al archivo CSV de salida.
    
    Returns:
    pd.DataFrame: DataFrame de pandas con el contenido del archivo CSV procesado.
    """
    columnas_a_eliminar = [
        "PERIOD", "UNITPRICE", "AMENITYID", "ISPARKINGSPACEINCLUDEDINPRICE",
        "PARKINGSPACEPRICE", "CONSTRUCTIONYEAR", "FLATLOCATIONID",
        "CADCONSTRUCTIONYEAR", "CADDWELLINGCOUNT", "geometry",
    ]

    columnas_a_convertir_entero = [
        "PRICE", "ROOMNUMBER", "BATHNUMBER", "FLOORCLEAN",
        "CADMAXBUILDINGFLOOR", "CADASTRALQUALITYID",
    ]

    atributos_a_normalizar = [
        "CONSTRUCTEDAREA", "ROOMNUMBER", "BATHNUMBER", "FLOORCLEAN",
        "CADMAXBUILDINGFLOOR", "CADASTRALQUALITYID", "ANTIQUITY",
        "DISTANCE_TO_CITY_CENTER", "DISTANCE_TO_METRO", "DISTANCE_TO_BLASCO"
    ]
    
    try:
        df = pd.read_csv(ruta_viviendas, delimiter=',', encoding='utf-8') # Viviendas
        gdf_barrios = gpd.read_file(ruta_barrios) # Barrios
        
        # Eliminacion de valores faltantes el la columna FLOORCLEAN
        df = df.dropna(subset=["FLOORCLEAN"])

        # Elimina duplicados en la columna ASSETID, conservando uno de forma aleatoria
        df = df.drop_duplicates(subset=["ASSETID"], keep="first")
        
        # Conversion de columnas numericas al tipo entero
        for columna in columnas_a_convertir_entero:
            if columna in df.columns:
                df[columna] = df[columna].fillna(0).astype(int)

        # Añade el atributo ANTIQUITY = 2018 - CADCONSTRUCTIONYEAR
        if "CADCONSTRUCTIONYEAR" in df.columns:
            df["ANTIQUITY"] = 2018 - df["CADCONSTRUCTIONYEAR"]
        
        # Elimina columnas innecesarias
        df = df.drop(columns=columnas_a_eliminar, errors='ignore')

        # Modificación de CADASTRALQUALITYID:
        if "CADASTRALQUALITYID" in df.columns:
            df["CADASTRALQUALITYID"] = df["CADASTRALQUALITYID"].apply(lambda x: 9 - x if pd.notnull(x) else x)

        def buscar_barrio(row):
            """
            La funcion determina en que barrio esta contenido
            la vivienda a partir de su latitud y longitud y el objeto
            geometrico de los barrios. Devolvera el barrio correspondiente o 
            'NA' si no se encuentra.

            Parametros:
            row (pd.Series): Fila del DataFrame que contiene la latitud y longitud.

            Returns:
            pd.Series: Barrio correspondiente o 'NA' si no se encuentra.
            """
            try:
                point = shapely.geometry.Point(float(row['LONGITUDE']), float(row['LATITUDE']))
                match = gdf_barrios[gdf_barrios.geometry.contains(point)]
                if not match.empty:
                    barrio = match.iloc[0]
                    return pd.Series([barrio['NEIGHBOURID']])
                else:
                    return pd.Series(['NA'])
            except Exception:
                return pd.Series(['NA'])

        df[['NEIGHBOURID']] = df.apply(buscar_barrio, axis=1)

        df = df[(df['NEIGHBOURID'] != 'NA')].reset_index(drop=True)

        # Guarda el DataFrame antes de normalizar
        if ruta_salida_no_normalizado:
            df.to_csv(ruta_salida_no_normalizado, index=False, encoding='utf-8')

        # Normalizacion
        scaler = MinMaxScaler()
        for col in atributos_a_normalizar:
            if col in df.columns:
                df[[col]] = scaler.fit_transform(df[[col]])

        df.to_csv(ruta_salida_viviendas, index=False, encoding='utf-8')

        # Elimina el archivo de entrada tras procesar
        if os.path.exists(ruta_viviendas):
            try:
                os.remove(ruta_viviendas)
                print(f"Archivo {ruta_viviendas} eliminado tras el procesamiento.")
            except Exception as e:
                print(f"No se pudo eliminar el archivo {ruta_viviendas}: {e}")

        print(f"El DataFrame tiene {len(df)} filas y {len(df.columns)} columnas despues del preprocesamiento.")
        return df
    except FileNotFoundError:
        print(f"El archivo {ruta_viviendas} no fue encontrado.")
    except Exception as e:
        print(f"Ocurrio un error al leer el archivo: {e}")
        return None

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    ruta_entrada = BASE_DIR / "raw_data" / "Valencia_Sale.csv"
    ruta_salida = BASE_DIR / "processed_data" / "Valencia_Sale_processed.csv"