from pathlib import Path
import pandas as pd
import csv
import geopandas as gpd
import shapely.geometry
import json
from sklearn.preprocessing import MinMaxScaler
import os

def procesar_csv_viviendas(csv_entrada):
    """
    Unifica la funcion de lectura y reescritura del archivo CSV en sucio.

    La funcion procesa un archivo CSV de entrada, seleccionando todas las columnas
    a excepcion de "geometry", la cual da problemas al leer el archivo con pandas.
    Luego, escribe el contenido en un nuevo archivo CSV de salida en la carpeta /processed_data.

    Parametros:
    csv_entrada (str): Ruta al archivo CSV de entrada.

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

    def escribir_csv(datos, encabezados, csv_entrada):
        """
        Funcion interna para escribir un archivo CSV a partir de una lista de diccionarios.
        Convierte la lista de diccionarios en un archivo CSV, donde cada diccionario representa una fila.
        El nombre de salida sera el del archivo de entrada + '_processed.csv', siempre en /processed_data.

        Parametros:
        datos (list): Lista de diccionarios con los datos a escribir.
        encabezados (list): Lista de encabezados para el archivo CSV.
        csv_entrada (str): Ruta al archivo CSV de entrada.

        Returns:
        Ninguno
        """
        try:
            base_dir = Path(__file__).resolve().parent
            directorio = base_dir / "processed_data"
            directorio.mkdir(parents=True, exist_ok=True)
            nombre_base = Path(csv_entrada).stem + "_processed.csv"
            ruta_salida_final = directorio / nombre_base

            with open(ruta_salida_final, mode='w', encoding='utf-8', newline='') as archivo:
                escritor = csv.DictWriter(archivo, fieldnames=encabezados)
                escritor.writeheader()  # Encabezados
                escritor.writerows(datos)  # Filas
            print(f"Archivo CSV creado correctamente en {ruta_salida_final}")
        except Exception as e:
            print(f"Ocurrio un error al escribir el archivo: {e}")

    # Ruta de salida
    base_dir = Path(__file__).resolve().parent
    directorio = base_dir / "processed_data"
    directorio.mkdir(parents=True, exist_ok=True)
    nombre_salida = Path(csv_entrada).name
    ruta_salida = directorio / nombre_salida

    contenido = leer_csv(csv_entrada)

    if contenido:
        encabezados = list(contenido[0].keys())
        escribir_csv(contenido, encabezados, csv_entrada)

def procesar_csv_barrios(ruta_csv):
    """
    Lee un CSV de barrios y realiza el preprocesamiento necesario para convertirlo
    en un GeoDataFrame. Elimina columnas innecesarias, renombra columnas y convierte
    la geometria de GeoJSON a objetos shapely.

    Guarda dos archivos de salida:
    - Un CSV con los barrios procesados para almacenar en la base de datos de Django.
    - Un GeoJSON con la geometria de los barrios.

    Parametros:
    ruta_csv (str): Ruta al archivo CSV de barrios.

    Returns:
    Ninguno
    """
    base_dir = Path(__file__).resolve().parent
    directorio = base_dir / "processed_data"
    directorio.mkdir(parents=True, exist_ok=True)

    # Nombres de salida
    nombre_base = Path(ruta_csv).stem + "_processed"
    ruta_csv_salida = directorio / "barris-barrios_data.csv"
    ruta_geojson_salida = directorio / f"{nombre_base}.geojson"

    df = pd.read_csv(ruta_csv, sep=';', dtype=str)

    # Columnas que no se necesitan
    df = df.drop(columns=['codbarrio', 'gis_gis_barrios_area', 'geo_point_2d', 'coddistrit'])

    # Se renombran las columnas para que siga el formato del CSV de viviendas
    df = df.rename(columns={
        'coddistbar': 'NEIGHBOURID',
        'nombre': 'NEIGHBOURNAME',
        'geo_shape': 'geometry'
    })

    # Identificadores unicos para cada barrio
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
    print(f"Archivos guardados en {ruta_csv_salida} y {ruta_geojson_salida}")

def preprocesamiento(ruta_viviendas, ruta_barrios):
    """
    Preprocesa el archivo CSV de entrada, eliminando atributos considerados
    como innecesarios para el contexto del problema, ademas de eliminar filas
    donde los valores sean nulos, vacios o incongruentes. Convierte ciertos
    atributos a tipos especificos.

    La funcion tambien normaliza ciertos atributos y guarda el resultado en un nuevo archivo CSV.

    Se generan dos archivos de salida:
    - Un CSV con los datos de viviendas preprocesados.
    - Un CSV con los datos de viviendas para almacenar en la base de datos de Django.

    Parametros:
    ruta_viviendas (str): Ruta al archivo CSV de viviendas.
    ruta_barrios (str): Ruta al archivo CSV de barrios.

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

        # Modificacion de CADASTRALQUALITYID:
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

        # Directorio de salida fijo y nombres de archivo requeridos
        base_dir = Path(__file__).resolve().parent
        processed_dir = base_dir / "processed_data"
        processed_dir.mkdir(parents=True, exist_ok=True)
        ruta_salida_viviendas = processed_dir / "Valencia_Sale_graph.csv"
        ruta_salida_no_normalizado = processed_dir / "Valencia_Sale_data.csv"

        # Guarda el DataFrame antes de normalizar
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