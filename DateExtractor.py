import zope.interface

class DateExtractor(zope.interface.Interface):

    def read_binary(self, filePath):
        pass

    def read_all(self, filePaths):
        pass

    def write_images(self, path, labels_images=[]):
        pass
