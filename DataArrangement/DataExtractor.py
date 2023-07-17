import zope.interface

class DataExtractor(zope.interface.Interface):

    def read_binary(self,filePaths,images_path,labels_images=[]):
        pass

    def read_all(self):
        pass

    def write_images(self):
        pass

    def extract_data(self):
        pass