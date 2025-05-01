from pathlib import Path
import pandas as pd
import csv

def procesar_csv(ruta_entrada, ruta_salida):
    """
    Unifica la funcion de lectura y reescritura del archivo CSV en sucio.

    La función procesa un archivo CSV de entrada, seleccionando todas las columnas
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
        Función interna para leer un archivo CSV en sucio y eliminar la columna "geometry".
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
                encabezados = encabezados[:41] # Como se ha indicado que la columna geometry es la 42, se eliminan todas las columnas a partir de la 41
                for fila in lector:
                    # Crea un diccionario para cada fila
                    vivienda = {encabezados[i]: fila[i] for i in range(len(encabezados))}
                    datos.append(vivienda)
        except FileNotFoundError:
            print(f"El archivo {ruta_archivo} no fue encontrado.") 
        except Exception as e:
            print(f"Ocurrió un error al leer el archivo: {e}")
        return datos

    def escribir_csv(ruta_salida, datos, encabezados):
        """
        Funcion interna para escribir un archivo CSV a partir de una lista de diccionarios.
        Convierte la lista de diccionarios en un archivo CSV, donde cada diccionario representa una fila.
        Devolverá un mensaje de éxito o error al escribir el archivo.

        Parametros:
        ruta_salida (str): Ruta al archivo CSV de salida, donde se guardará el archivo.
        datos (list): Lista de diccionarios con los datos a escribir.
        encabezados (list): Lista de encabezados para el archivo CSV.

        Returns:
        Ninguno
        """
        try:
            with open(ruta_salida, mode='w', encoding='utf-8', newline='') as archivo:
                escritor = csv.DictWriter(archivo, fieldnames=encabezados)
                escritor.writeheader()  # Escribir los encabezados
                escritor.writerows(datos)  # Escribir las filas
            print(f"Archivo CSV creado correctamente en /processed_data")
        except Exception as e:
            print(f"Ocurrió un error al escribir el archivo: {e}")

    # Crea la carpeta de salida si no existe
    Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)

    # Lee los datos del archivo CSV de entrada
    contenido = leer_csv(ruta_entrada)

    # Escribe los datos
    if contenido:
        encabezados = list(contenido[0].keys())  # Encabezados del CSV
        escribir_csv(ruta_salida, contenido, encabezados)

def preprocesamiento_iteracion1(ruta_archivo, ruta_salida):
    """
    Preprocesa el archivo CSV de entrada, eliminando atributos considerados
    como innecesarios para el contexto del problema, además de eliminar filas
    donde los valores sean nulos, vacíos o incongruentes. Convierte ciertos
    atributos a tipos específicos. Unifica BUILTTYPEID_1, BUILTTYPEID_2 
    y BUILTTYPEID_3 en un único atributo STATUS.

    Parametros:
    ruta_archivo (str): Ruta al archivo CSV.
    ruta_salida (str): Ruta al archivo CSV de salida.
    
    Returns:
    pd.DataFrame: DataFrame de pandas con el contenido del archivo CSV procesado.
    """
    columnas_a_eliminar = [
        "PERIOD", "AMENITYID", "ISPARKINGSPACEINCLUDEDINPRICE",
        "PARKINGSPACEPRICE", "CONSTRUCTIONYEAR", "FLATLOCATIONID",
        "CADCONSTRUCTIONYEAR", "CADDWELLINGCOUNT", "BUILTTYPEID_1", 
        "BUILTTYPEID_2", "BUILTTYPEID_3", "geometry",
    ]

    columnas_a_convertir_booleano = [
        "HASTERRACE", "HASLIFT", "HASAIRCONDITIONING", "HASPARKINGSPACE", 
        "HASNORTHORIENTATION", "HASSOUTHORIENTATION", "HASEASTORIENTATION", 
        "HASWESTORIENTATION", "HASBOXROOM", "HASWARDROBE", "HASSWIMMINGPOOL", 
        "HASGARDEN", "HASDOORMAN", "ISDUPLEX", "ISSTUDIO", "ISINTOPFLOOR",
    ]

    columnas_a_convertir_entero = [
        "PRICE", "ROOMNUMBER", "BATHNUMBER", "FLOORCLEAN",
        "CADMAXBUILDINGFLOOR", "CADASTRALQUALITYID"
    ]
    
    try:
        # Lee el archivo CSV con pandas
        df = pd.read_csv(ruta_archivo, delimiter=',', encoding='utf-8')
        
        # Unifica BUILTTYPE_1, BUILTTYPE_2 y BUILTTYPE_3 en STATUS
        def determinar_status(row):
            if row.get("BUILTTYPEID_1") == 1:
                return "NEWCONSTRUCTION"
            elif row.get("BUILTTYPEID_2") == 1:
                return "2HANDRESTORE"
            elif row.get("BUILTTYPEID_3") == 1:
                return "2HANDGOOD"
            return None

        df["STATUS"] = df.apply(determinar_status, axis=1)

        # Elimina las columnas especificadas en la lista
        df = df.drop(columns=columnas_a_eliminar, errors='ignore')
        
        # Elimina filas donde "FLOORCLEAN" tiene valor NA
        df = df.dropna(subset=["FLOORCLEAN"])

        # Elimina duplicados en la columna ASSETID, conservando uno de forma aleatoria
        df = df.drop_duplicates(subset=["ASSETID"], keep="first")
        
        # Convierte las columnas especificadas de 0/1 a booleanos
        for columna in columnas_a_convertir_booleano:
            if columna in df.columns:
                df[columna] = df[columna].astype(bool)
        
        # Convierte las columnas especificadas a enteros
        for columna in columnas_a_convertir_entero:
            if columna in df.columns:
                df[columna] = df[columna].fillna(0).astype(int)

        # Guarda el DataFrame procesado en un nuevo archivo CSV
        df.to_csv(ruta_salida, index=False, encoding='utf-8')

        print(f"El DataFrame tiene {len(df)} filas y {len(df.columns)} columnas después del preprocesamiento.")
        return df
    except FileNotFoundError:
        print(f"El archivo {ruta_archivo} no fue encontrado.")
    except Exception as e:
        print(f"Ocurrió un error al leer el archivo: {e}")
        return None

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    ruta_entrada = BASE_DIR / "raw_data" / "Valencia_Sale.csv"
    ruta_salida = BASE_DIR / "processed_data" / "Valencia_Sale_processed.csv"
    ruta_salida_pandas = BASE_DIR / "processed_data" / "Valencia_Sale_it1.csv"
    procesar_csv(ruta_entrada, ruta_salida)
    preprocesamiento_iteracion1(ruta_salida, ruta_salida_pandas)