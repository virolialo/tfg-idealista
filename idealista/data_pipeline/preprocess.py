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

        Raises:
        FileNotFoundError: Si el archivo no existe.
        Exception: Si ocurre un error al leer el archivo.
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
        Ninguno.

        Raises:
        Exception: Si ocurre un error al escribir el archivo.
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

    contenido = leer_csv(csv_entrada)

    if contenido:
        encabezados = list(contenido[0].keys())
        escribir_csv(contenido, encabezados, csv_entrada)

def procesar_csv_barrios(ruta_csv):
    """
    La funcion procesa un archivo CSV de barrios donde:
    - Elimina columnas innecesarias.
    - Renombra columnas para estandarizar.
    - Asegura que la columna NEIGHBOURID sea unica y numerica.

    Genera dos archivos de salida:
    - Un CSV simple para Django con las columnas NEIGHBOURID y NEIGHBOURNAME.
    - Un GeoJSON con la geometria de los barrios.

    Parametros:
    ruta_csv (str): Ruta al archivo CSV de barrios.

    Returns:
    Ninguno.

    Raises:
    Exception: Si ocurre un error durante el procesamiento del CSV.
    """
    try:
        base_dir = Path(__file__).resolve().parent
        processed_dir = base_dir / "processed_data"
        processed_dir.mkdir(parents=True, exist_ok=True)

        nombre_base = Path(ruta_csv).stem + "_processed"
        ruta_csv_salida = processed_dir / "barris-barrios_data.csv"
        ruta_geojson_salida = processed_dir / f"{nombre_base}.geojson"

        df = pd.read_csv(ruta_csv, sep=';', dtype=str) # Barrios

        # Eliminacion de columnas
        columnas_a_eliminar = ['codbarrio', 'gis_gis_barrios_area', 'geo_point_2d', 'coddistrit']
        df = df.drop(columns=columnas_a_eliminar, errors='ignore')

        # Estandarizacion de cabeceras
        df = df.rename(columns={
            'coddistbar': 'NEIGHBOURID',
            'nombre': 'NEIGHBOURNAME',
            'geo_shape': 'geometry'
        })

        # Identificador unico
        if not pd.api.types.is_numeric_dtype(df['NEIGHBOURID']):
            df['NEIGHBOURID'] = range(1, len(df) + 1)

        # CSV de datos para Django
        df[['NEIGHBOURID', 'NEIGHBOURNAME']].to_csv(ruta_csv_salida, index=False, encoding='utf-8')

        # Convertir la columna geometry (GeoJSON string) a objetos shapely
        def geojson_to_shape(geojson_str):
            """
            La funcion convierte una cadena GeoJSON a un objeto shapely.

            Parametros:
            geojson_str (str): Cadena GeoJSON que representa la geometria.

            Returns:
            shapely.geometry: Objeto shapely representando la geometria.
            None: Si la conversion falla.

            Raises:
            Exception: Si ocurre un error al convertir la cadena GeoJSON.
            """
            try:
                geojson = json.loads(geojson_str)
                return shapely.geometry.shape(geojson)
            except Exception:
                return None

        df['geometry'] = df['geometry'].apply(geojson_to_shape)

        # GeoJSON de barrios
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
        gdf.to_file(ruta_geojson_salida, driver='GeoJSON')
        print(f"Archivos guardados en {ruta_csv_salida} y {ruta_geojson_salida}")

    except Exception as e:
        print(f"Error procesando el CSV de barrios: {e}")

def preprocesamiento(ruta_viviendas, ruta_barrios):
    """
    La funcion realiza el preprocesamiento de los datos de viviendas y barrios, para
    preparar los datos para su uso en modelos de machine learning.

    Genera los siguientes archivos de salida:
    - Un CSV con los datos de viviendas preprocesados.
    - Un CSV con los datos de viviendas sin normalizar.

    Modifica/elimina los siguientes archivos:
    - Un GeoJSON con los barrios que tienen viviendas asociadas.
    - Un CSV con los barrios que tienen viviendas asociadas.
    - Elimina el archivo de entrada de viviendas tras el procesamiento.

    Parametros:
    ruta_viviendas (str): Ruta al archivo CSV de viviendas.
    ruta_barrios (str): Ruta al archivo GeoJSON de barrios.

    Returns:
    pd.DataFrame: DataFrame con los datos de viviendas preprocesados.
    None: Si ocurre un error durante el procesamiento.

    Raises:
    FileNotFoundError: Si el archivo de viviendas no se encuentra.
    Exception: Si ocurre un error al leer o procesar los archivos.
    """
    columnas_a_eliminar = [
        "PERIOD", "PRICE", "AMENITYID", "ISPARKINGSPACEINCLUDEDINPRICE",
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
        
        # Elimina de valores faltantes de FLOORCLEAN
        df = df.dropna(subset=["FLOORCLEAN"])

        # Elimina duplicados en ASSETID
        df = df.drop_duplicates(subset=["ASSETID"], keep="first") # Conserva el primero
        
        # Conversion columnas numericas a entero
        for columna in columnas_a_convertir_entero:
            if columna in df.columns:
                df[columna] = df[columna].fillna(0).astype(int)

        # Añade atributo ANTIQUITY = 2018 - CADCONSTRUCTIONYEAR
        if "CADCONSTRUCTIONYEAR" in df.columns:
            df["ANTIQUITY"] = 2018 - df["CADCONSTRUCTIONYEAR"]
        
        # Elimina columnas innecesarias
        df = df.drop(columns=columnas_a_eliminar, errors='ignore')

        # Modifica CADASTRALQUALITYID:
        if "CADASTRALQUALITYID" in df.columns:
            df["CADASTRALQUALITYID"] = df["CADASTRALQUALITYID"].apply(lambda x: 9 - x if pd.notnull(x) else x)

        def buscar_barrio(row):
            """
            La funcion busca el barrio correspondiente a una vivienda
            utilizando las coordenadas de latitud y longitud.
            Si no encuentra un barrio, devuelve 'NA'.

            Parametros:
            row (pd.Series): Fila del DataFrame con las coordenadas de la vivienda.

            Returns:
            pd.Series: Serie con el NEIGHBOURID del barrio encontrado o 'NA'.

            Raises:
            Exception: Si ocurre un error al crear el punto o buscar el barrio.
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

        # Directorio y archivos de salida
        base_dir = Path(__file__).resolve().parent
        processed_dir = base_dir / "processed_data"
        processed_dir.mkdir(parents=True, exist_ok=True)
        ruta_salida_viviendas = processed_dir / "Valencia_Sale_graph.csv"
        ruta_salida_no_normalizado = processed_dir / "Valencia_Sale_data.csv"
        ruta_salida_barrios = processed_dir / "barris-barrios_data.csv"

        # CSV no normalizado
        df.to_csv(ruta_salida_no_normalizado, index=False, encoding='utf-8')

        # Normalizacion
        scaler = MinMaxScaler()
        for col in atributos_a_normalizar:
            if col in df.columns:
                df[[col]] = scaler.fit_transform(df[[col]])

        # CSV normalizado
        df.to_csv(ruta_salida_viviendas, index=False, encoding='utf-8')

        # Elimina barrios sin viviendas asociadas
        if ruta_salida_barrios.exists():
            barrios_df = pd.read_csv(ruta_salida_barrios, encoding='utf-8')
            barrios_con_viviendas = df['NEIGHBOURID'].unique()
            barrios_df_filtrado = barrios_df[barrios_df['NEIGHBOURID'].astype(str).isin(barrios_con_viviendas.astype(str))]
            barrios_df_filtrado.to_csv(ruta_salida_barrios, index=False, encoding='utf-8')
            print(f"Barrios sin viviendas eliminados de {ruta_salida_barrios}")

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
    ruta_viviendas = BASE_DIR / "raw_data" / "Valencia_Sale.csv"
    ruta_barrios = BASE_DIR / "raw_data" / "barris-barrios.csv"
    procesar_csv_viviendas(ruta_viviendas)
    procesar_csv_barrios(ruta_barrios)
    viviendas = BASE_DIR / "processed_data" / "Valencia_Sale_processed.csv"
    barrios = BASE_DIR / "processed_data" / "barris-barrios_processed.geojson"
    preprocesamiento(viviendas, barrios)
    print("Preprocesamiento completado.")