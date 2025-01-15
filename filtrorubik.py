import skimage.io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans

class FiltroRubik:
    """
    Clase para aplicar filtros tipo Rubik a imágenes, extraer paletas de colores y manipular imágenes.

    Atributos:
    ----------
    _paletas : dict
        Diccionario que contiene paletas de colores predeterminadas.
    _imagen_modificada : ndarray
        Imagen modificada después de aplicar el filtro.
    """

    def __init__(self):
        """
        Inicializa la clase con una lista de paletas de colores predeterminadas y una imagen vacia.
        """
        self._paletas = {
            "paleta1": ["#FF0000", "#0000FF", "#FFFF00", "#FFFFFF", "#FFAF00", "#00FF00"],
            "paleta2": ["#98A2EB", "#98EBAD", "#EA999D", "#EBD898", "#967A7C", "#464A6B", "#FFFFFF"],
            "paleta3": ["#D9296A", "#BF2A97", "#C21AD9", "#532259", "#262240"]
        }
        self._imagen_modificada = None


    def _reescale(self, nombre_img, cuadro_size):
        """
        Rescala la imagen de entrada a un tamaño que sea múltiplo de `cuadro_size`.

        Parameters:
        -----------
        nombre_img : str
            La ruta al archivo de imagen de entrada.
        cuadro_size : int
            El tamaño deseado de cada cuadro en la imagen reescalada.

        Returns:
        --------
        ndarray
            La imagen reescalada, con dimensiones que son múltiplos de `cuadro_size`.
        """
        imagen = skimage.io.imread(nombre_img)
        return imagen[:imagen.shape[0] // cuadro_size * cuadro_size, :imagen.shape[1] // cuadro_size * cuadro_size]


    def _hex_to_rgb(self, hex_colores):
        """
        Convierte una lista de colores en formato hexadecimal a formato RGB.

        Parameters:
        -----------
        hex_colores : list
            Lista de colores en formato hexadecimal.

        Returns
        --------
        list
            Lista de colores en formato RGB.
        """
        rgb_colors = []
        for hex_color in hex_colores:
            hex_color = hex_color.lstrip('#')
            rgb_color = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
            rgb_colors.append(rgb_color)
        return rgb_colors


    def _rgb_to_hex(self, rgb_colors):
        """
        Convierte una lista de colores en formato RGB a formato hexadecimal.

        Parameters:
        -----------
        rgb_colors : list
            Lista de colores en formato RGB.

        Returns:
        --------
        list
            Lista de colores en formato hexadecimal.
        """
        hex_colors = []
        for rgb_color in rgb_colors:
            hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb_color)
            hex_color = hex_color.upper()
            hex_colors.append(hex_color)
        return hex_colors
    

    def _png_a_jpg(self, ruta_png):
        """
        Convierte una imagen PNG a formato JPG.

        Parameters:
        -----------
        ruta_png : str
            La ruta al archivo de imagen PNG.

        Returns:
        --------
        str
            La ruta al archivo de imagen JPG convertido.
        """
        imagen_png = Image.open(ruta_png)
        imagen_rgb = imagen_png.convert('RGB')
        ruta_jpg = ruta_png.replace('.png', '.jpg')
        #imagen_rgb.save(ruta_jpg, 'JPEG')
        return ruta_jpg


    def _distancia_color(self, color1, color2):
        """
        Calcula la distancia euclidiana entre dos colores.

        Parameters:
        -----------
        color1 : list
            Primer color en formato RGB.
        color2 : list
            Segundo color en formato RGB.

        Returns:
        --------
        float
            La distancia euclidiana entre los dos colores.
        """
        return np.linalg.norm(np.array(color1) - np.array(color2))


    def _color_cercano(self, color, colores):
        """
        Encuentra el color más cercano a un color dado en una lista de colores.

        Paramaters:
        -----------
        color : list
            Color en formato RGB.
        colores : list
            Lista de colores en formato RGB.

        Returns:
        --------
        list
            El color más cercano en formato RGB.
        """
        copia_colores = colores.copy()
        copia_colores = sorted(copia_colores, key=lambda c: self._distancia_color(color, c), reverse=False)
        if np.random.rand() < 0.95:
            return copia_colores[0]
        else:
            return copia_colores[1]


    def _extraer_paleta(self, nombre_imagen, len_paleta):
        """
        Extrae una paleta de colores de una imagen utilizando el algoritmo KMeans.

        Parameters:
        -----------
        nombre_imagen : str
            La ruta al archivo de imagen.
        len_paleta : int
            El número de colores en la paleta.

        Returns:
        --------
        list
            Lista de colores en formato RGB.
        """
        img = skimage.io.imread(nombre_imagen)
        imagen_modificada = img.copy()

        # Método KMeans
        kmeans = KMeans(n_clusters= len_paleta, random_state=14)
        kmeans.fit(imagen_modificada.reshape(-1, 3))
        centroids = [list(map(int, c)) for c in kmeans.cluster_centers_]
        return centroids


    def obtener_paleta(self, imagen, len_paleta):
        """
        Extrae una paleta de colores de una imagen y la convierte a formato hexadecimal.

        Parameterx:
        -----------
        imagen : str
            La ruta al archivo de imagen.
        len_paleta : int
            El número de colores en la paleta.

        Returns:
        --------
        list
            Lista de colores en formato hexadecimal.
        """
        if imagen.endswith('.png'):
            imagen = self._png_a_jpg(imagen)
        colores = self._extraer_paleta(imagen, len_paleta)
        paleta = self._rgb_to_hex(colores)
        return paleta


    def plot_paleta(self, imagen, len_paleta):
        """
        Muestra una paleta de colores extraída de una imagen.

        Parameters:
        -----------
        imagen : str
            La ruta al archivo de imagen.
        len_paleta : int
            El número de colores en la paleta.

        Returns:
        ------
        ValueError
            Si `len_paleta` es menor o igual a 1.
        """
        # Manejo de excepciones
        if len_paleta <= 1:
            raise ValueError("Se necesitan al menos dos colores")

        if imagen.endswith('.png'):
            imagen = self._png_a_jpg(imagen)

        colores_paleta = self._extraer_paleta(imagen, len_paleta)
        colores_hex = self._rgb_to_hex(colores_paleta)

        n_colores = len(colores_paleta)

        # Crear figura con subplots en una sola fila y tantas columnas como colores
        fig, axs = plt.subplots(1, n_colores, figsize=(n_colores * 2, 2))

        for i, ax in enumerate(axs):
            ax.imshow(np.full((10, 10, 3), colores_paleta[i], dtype=np.uint8))
            ax.axis('off')
            ax.set_title(colores_hex[i])
        plt.show()


    def transferir_paleta(self, imagen1, imagen2, cuadro_size, len_paleta):
        """
        Aplica un filtro a una imagen utilizando una paleta de colores extraída de otra imagen.

        Parameters:
        -----------
        imagen1 : str
            La ruta al archivo de la imagen a la que se aplicará el filtro.
        imagen2 : str
            La ruta al archivo de la imagen de la que se extraerá la paleta de colores.
        cuadro_size : int
            El tamaño de cada cuadro en la imagen filtrada.
        len_paleta : int
            El número de colores en la paleta.
        """
        if imagen1.endswith('.png'):
            imagen1 = self._png_a_jpg(imagen1)
        if imagen2.endswith('.png'):
            imagen2 = self._png_a_jpg(imagen2)
        paleta_robada = self.obtener_paleta(imagen2, len_paleta)
        self.filtro(imagen1, cuadro_size, paleta_robada)
    

    def filtro(self, imagen, cuadro_size, paleta=None):
        """
        Aplica un filtro tipo Rubik a una imagen.

        Parameters:
        -----------
        imagen : str
            La ruta al archivo de imagen.
        cuadro_size : int
            El tamaño de cada cuadro en la imagen filtrada.
        paleta : list or str, opcional
            La paleta de colores a utilizar. Puede ser una lista de colores en formato hexadecimal o el nombre de una paleta predeterminada.

        Returns:
        ------
        ValueError
            Si `paleta` no es una lista o una cadena.
        """
        if imagen.endswith('.png'):
            imagen = self._png_a_jpg(imagen)

        imagen = self._reescale(imagen, cuadro_size)
        
        # Paleta por default si no se asigna ninguna
        if paleta is None:
            lista_colores_hex = self._paletas['paleta1']
        # Asignar alguna de las paletas de la clase
        elif isinstance(paleta, str):
            lista_colores_hex = self._paletas[paleta]
        # Usar paleta si se da una lista de colores
        elif isinstance(paleta, list):
            lista_colores_hex = paleta
        else:
            raise ValueError("El argumento 'paleta' debe ser una lista o una cadena.")
        
        lista_colores = self._hex_to_rgb(lista_colores_hex)
        imagen_modificada = imagen.copy()

        for i in range(0, imagen_modificada.shape[0], cuadro_size):
            for j in range(0, imagen_modificada.shape[1], cuadro_size):
                cuadro = imagen_modificada[i:i+cuadro_size, j:j+cuadro_size]
                media = np.mean(cuadro, axis=(0, 1))
                color_cercano_media = self._color_cercano(media, lista_colores)
                imagen_modificada[i:i+cuadro_size, j:j+cuadro_size] = color_cercano_media
        
        self._imagen_modificada = imagen_modificada


    def show(self):
        """
        Muestra la imagen modificada.

        Returns:
        ------
        ValueError
            Si no se ha aplicado ningún filtro.
        """
        if self._imagen_modificada is not None:
            skimage.io.imshow(self._imagen_modificada)
            plt.show()
        else:
            raise ValueError("No se ha aplicado ningún filtro. Use el método 'filtro' primero.")


    def guardar(self, filename=None):
            """
            Guarda la imagen modificada en un archivo.

            Parameters:
            - filename: str, opcional. El nombre del archivo en el que se guardará la imagen modificada.
                        Si no se proporciona un nombre, se utilizará 'imagen_modificada.jpg' por defecto.
            """
            if self._imagen_modificada is None:
                raise ValueError("No se ha aplicado ningún filtro. Use el método 'filtro' primero.")
            if filename is None:
                filename = 'imagen_modificada.jpg'  # Si no se agrega un nombre a la imagen, nombrarla por default
            skimage.io.imsave(filename, self._imagen_modificada)
