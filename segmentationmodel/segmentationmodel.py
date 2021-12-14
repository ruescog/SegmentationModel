# FUNCIONES BASICAS PARA LA GESTION DE LA LIBRERIA
class Logger():
    def __init__(self):
        self.inicio = time.time()

    def log(self, mensaje):
        print(f"[{round(time.time() - self.inicio, 2)}s]: {mensaje}")

def get_mmseg():
    """
    Actualiza la versión de MMSEG y configura los archivos para poder utilizar los modelos de esta librería.
    """

    # coge la ultima version estable del repo
    response = get("https://api.github.com/repos/open-mmlab/mmsegmentation/releases/latest")

    if response.ok:
        # recoge el tag para poder clonarla
        release = response.json()["tag_name"]
    else:
        LOG.log("URL no encontrada.")
        return
    
    # si existen archivos en local, puede ser necesario actualizarlos
    if os.path.isdir("mmseg"):
        # debe actualizar?
        with open("mmseg/configs/_version/version.txt", "r") as f:
            release_antigua = f.read()

        if release == release_antigua:
            # no hay que actualizar
            return
        else:
            # hay que actualizar
            LOG.log("(*) Actualizando MMSEGMENTATION.")
            LOG.log("| Borrando los archivos antiguos.")
            rmtree("mmseg")
            
    else:
        # debe instalar
        LOG.log("(*) Instalando MMSEGMENTATION:")
    
    # clona el repositorio: desde - a - rama
    LOG.log("| Descargando los archivos necesarios.")
    Repo.clone_from("https://github.com/open-mmlab/mmsegmentation", "./mmsegmentation", branch=release)

    # modifica la forma en la que se almacenan los ficheros
    LOG.log("| Configurando los directorios de archivos.")
    os.rename("mmsegmentation/mmseg", "mmseg")
    os.rename("mmsegmentation/configs", "mmseg/configs")
    os.mkdir("mmseg/configs/_version")
    rmtree("mmsegmentation")
    with open("mmseg/configs/_version/version.txt", "w") as f:
        f.write(release)

    LOG.log("(*) MMSEGMENTATION instalado correctamente.")


# IMPORTACION DE LOS PAQUETES NECESARIOS
import os # gestion de librerias desde el so
from shutil import rmtree # función para poder borrar recursivamente un directorio
from requests import get # función para poder hacer una petición get.
import yaml # gestor de archivos yaml
import time # gestion del tiempo
import warnings # para la gestion de warnings
from git import Repo # clonado de repos git
import torch # torch para poder elegir si usar cpu o gpu

# CONFIGURACION GENERAL DE PAQUETES Y VARIABLES
LOG = Logger() # variable para la gestion de mensajes
warnings.filterwarnings("ignore")

# comprueba las versiones de mmseg y actualiza si es necesario
get_mmseg()

# librerias abstraidas
import semtorch # libreria semtorch
import segmentron # libreria segmentron
import segmentation_models_pytorch as smp # libreria segmentation_models_pytroch
from mmseg.apis import init_segmentor # libreria mmsegmentation


# OBJETOS DE LA FACHADA
class SegmentationModel():
    """
    Objeto genérico que crea una fachada sobre los objetos Learner y EncoderDecoder
    """
    def __init__(self, libreria, modelo, arquitectura, esqueleto, pesos):
        self.libreria = libreria
        self.modelo = modelo
        self.arquitectura = arquitectura
        self.esqueleto = esqueleto
        self.pesos = pesos

    def learning_rate(self):
        self.modelo.lr_find()
        self.modelo.learn.recorder

    def entrenar(self, **kwargs):
        """
        Aplica la función fit_one_cycle al modelo.
        """
        
        if self.libreria == "mmsegmentation":
            # TODO entrenar un modelo para las librerías ...
            pass
        else:
            self.modelo.fit_one_cycle(**kwargs)


class SegmentationManager():
    """
    Gestor de la fachada sobre la creación de los modelos en todas las librerías.
    """
    def _elegir_libreria(arquitectura, esqueleto, pesos=None):
        """
        Devuelve la librería indicada para la elección de un modelo.
        """
        # TODO ¿estos archivos deberían estar en local?

        if pesos:
            # carga los ficheros de configuracion
            modelos_smp = get("https://raw.githubusercontent.com/ruescog/SegmentationModel/main/smp.config.json").json()
            hay_arquitectura = False
            hay_esqueleto = False
            # busca el mejor modelo para los modelos con pesos
            if arquitectura in modelos_smp["arquitecturas"].split(" "):
                hay_arquitectura = True
            if esqueleto in modelos_smp["esqueletos"]:
                hay_esqueleto = True
            if hay_arquitectura and hay_esqueleto and pesos in modelos_smp["esqueletos"][esqueleto].split(" "):
                return "smp"
            else:
                if not hay_arquitectura:
                    raise AttributeError(
                        f"Arquitectura '{arquitectura}', esqueleto '{esqueleto}' o pesos '{pesos}' no encontrados.")
                elif not hay_esqueleto:
                    raise AttributeError(
                        f"Esqueleto '{esqueleto}' no encontrado. Los válidos son '{list(modelos_smp['esqueletos'].keys())}'.")
                else:
                    raise AttributeError(
                        f"Pesos '{pesos}' no encontrados. Los válidos para '{esqueleto}' son '{modelos_smp['esqueletos'][esqueleto]}'")
        elif esqueleto[-3:] == ".py":
            arquitecturas = list(filter(lambda nombre: "_" not in nombre, os.listdir("mmseg/configs")))
            if arquitectura in arquitecturas:
                return "mmsegmentation"
            else:
                raise AttributeError(
                    f"Arquitectura '{arquitectura}' no encontrada. Las válidas son {arquitecturas}.")
        else:
            modelos_semtorch = get("https://raw.githubusercontent.com/ruescog/SegmentationModel/main/semtorch.config.json").json()
            modelos_segmentron = get("https://raw.githubusercontent.com/ruescog/SegmentationModel/main/segmentron.config.json").json()
            modelos = {"semtorch": modelos_semtorch, "segmentron": modelos_segmentron}
            hay_arquitectura = False

            # busca el mejor modelo para los modelos sin pesos
            for libreria_nombre in modelos:
                libreria = modelos[libreria_nombre]
                if arquitectura in libreria:
                    hay_arquitectura = True
                if esqueleto in libreria[arquitectura].split(" "):
                    return libreria_nombre
            else:
                if not hay_arquitectura:
                    raise AttributeError(
                        f"Arquitectura '{arquitectura}' o esqueleto '{esqueleto}' no encontrados.")
                else:
                    raise AttributeError(
                        f"Esqueleto '{esqueleto}' no encontrado. Los válidos para '{arquitectura}' son '{libreria[arquitectura]}'.")

    def _buscar_ficheros(arquitectura, esqueleto):
        """
        Devuelve la ruta del esqueleto y de los pesos para MMSEGMENTATION.
        """
        
        ruta = f"mmseg/configs/{arquitectura}/"
        with open(ruta + arquitectura + ".yml") as fichero:
            configuracion = yaml.load(fichero, Loader=yaml.FullLoader)
            
        esqueletos = []
        for modelo in configuracion["Models"]:
            esqueletos.append(modelo["Name"])
            if modelo["Name"] == esqueleto[:-3]:
                return ruta + esqueleto, modelo["Weights"]
        else:
            raise AttributeError(
                f"Esqueleto '{esqueleto}' no encontrado para la arquitectura '{arquitectura}'. Los válidos son {esqueletos}.")

    def crear_modelo(arquitectura, esqueleto, pesos=None, dataloader=None, num_clases=2, **kwargs):
        """
        Crea un modelo de segmentación de imagen con las características solicitadas.
        :param arquitectura: La arquitectura que usará el modelo.
        :type arquitectura: str
        :param esqueleto: El esqueleto de la arquitectura del modelo.
        :type esqueleto: str
        :param pesos: Opcional. Los pesos con los que se ha entrenado el esqueleto.
        :type pesos: str or None
        :param dataloader: Opcional. Conjunto de datos de entrenamiento para crear el modelo.
        :type dataloader: DataLoader or None
        :param num_clases: Por defecto, 2. Número de clases del modelo.
        :type num_clases: int
        :param **kwargs**: Otros parámetros de construcción del modelo.
        :tpye **kwargs: dict
        """
        
        libreria = SegmentationManager._elegir_libreria(
            arquitectura, esqueleto, pesos)

        # para la libreria semtorch
        if libreria == "semtorch":
            modelo = semtorch.get_segmentation_learner(dls=dataloader, number_classes=num_clases, segmentation_type="Semantic Segmentation",
                                                       architecture_name=arquitectura, backbone_name=esqueleto, **kwargs).to_fp16()

        # para la libreria segmentron
        elif libreria == "segmentron":
            if arquitectura == "bisenet":
                modelo = segmentron.BiSeNet(
                    backbone_name=esqueleto, nclass=num_clases)
            elif arquitectura == "cgnet":
                modelo = segmentron.CGNet(nclass=num_clases)
            elif arquitectura == "contextnet":
                modelo = segmentron.ContextNet(nclass=num_clases)
            elif arquitectura == "dabnet":
                # FIXME implementar este modelo
                raise NotImplementedError()
            elif arquitectura == "deeplabv3_plus":
                modelo = segmentron.DeepLabV3Plus(
                    nclass=num_clases, backbone_name=esqueleto)
            elif arquitectura == "denseaspp":
                modelo = segmentron.DenseASPP(
                    nclass=num_clases, backbone_name=esqueleto)
            elif arquitectura == "dfanet":
                # FIXME implementar este modelo
                raise NotImplementedError()
            elif arquitectura == "fcn":
                modelo = segmentron.FCN(
                    nclass=num_clases, backbone_name=esqueleto)
            elif arquitectura == "fpenet":
                modelo = segmentron.FPENet(nclass=num_clases)
            elif arquitectura == "hrnet":
                modelo = segmentron.HRNet(
                    nclass=num_clases, backbone_name=esqueleto)
            elif arquitectura == "lednet":
                modelo = segmentron.LEDNet(nclass=num_clases)
            elif arquitectura == "ocnet":
                modelo = segmentron.OCNet(
                    nclass=num_clases, backbone_name=esqueleto)
            elif arquitectura == "pspnet":
                modelo = segmentron.PSPNet(
                    nclass=num_clases, backbone_name=esqueleto)
            elif arquitectura == "unet":
                modelo = segmentron.UNet(nclass=num_clases)

            modelo = Learner(dataloader, modelo, **kwargs).to_fp16()

        # para la libreria smp
        elif libreria == "smp":
            if arquitectura == "deeplabv3":
                modelo = smp.DeepLabV3(
                    encoder_name=esqueleto, encoder_weights=pesos, classes=num_clases)
            elif arquitectura == "deeplabv3plus":
                modelo = smp.DeepLabV3Plus(
                    encoder_name=esqueleto, encoder_weights=pesos, classes=num_clases)
            elif arquitectura == "fpn":
                modelo = smp.FPN(encoder_name=esqueleto,
                                 encoder_weights=pesos, classes=num_clases)
            elif arquitectura == "linknet":
                modelo = smp.Linknet(encoder_name=esqueleto,
                                     encoder_weights=pesos, classes=num_clases)
            elif arquitectura == "manet":
                modelo = smp.MAnet(encoder_name=esqueleto,
                                   encoder_weights=pesos, classes=num_clases)
            elif arquitectura == "pan":
                modelo = smp.PAN(encoder_name=esqueleto,
                                 encoder_weights=pesos, classes=num_clases)
            elif arquitectura == "pspnet":
                modelo = smp.PSPNet(encoder_name=esqueleto,
                                    encoder_weights=pesos, classes=num_clases)
            elif arquitectura == "unet":
                modelo = smp.Unet(encoder_name=esqueleto,
                                  encoder_weights=pesos, classes=num_clases)
            elif arquitectura == "unetplusplus":
                modelo = smp.UnetPlusPlus(
                    encoder_name=esqueleto, encoder_weights=pesos, classes=num_clases)

            modelo = Learner(dataloader, modelo, **kwargs).to_fp16()

        # para la libreria mmsegmentation
        elif libreria == "mmsegmentation":
            # hay que buscar la ruta completa del archivo introducido por el usuario
            esqueleto, pesos = SegmentationManager._buscar_ficheros(arquitectura, esqueleto)
            servicio = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            modelo = init_segmentor(esqueleto, pesos, device=servicio)

        return SegmentationModel(libreria, modelo, arquitectura, esqueleto, pesos)