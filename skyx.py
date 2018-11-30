''' Module to handle connections to TheSkyX
The classes are defined to match the classes in Script TheSkyX. This isn't
really necessary as they all just send the javascript to TheSkyX via
SkyXConnection._send().
'''
from __future__ import print_function

import logging
import time
from socket import socket, AF_INET, SOCK_STREAM, SHUT_RDWR, error


logger = logging.getLogger(__name__)

class Singleton(object):
    ''' Singleton class so we dont have to keep specifing host and port'''
    def __init__(self, klass):
        ''' Initiator '''
        self.klass = klass
        self.instance = None

    def __call__(self, *args, **kwds):
        ''' When called as a function return our singleton instance. '''
        if self.instance is None:
            self.instance = self.klass(*args, **kwds)
        return self.instance


class SkyxObjectNotFoundError(Exception):
    ''' Exception for objects not found in SkyX.
    '''
    def __init__(self, value):
        ''' init'''
        super(SkyxObjectNotFoundError, self).__init__(value)
        self.value = value

    def __str__(self):
        ''' returns the error string '''
        return repr(self.value)


class SkyxConnectionError(Exception):
    ''' Exception for Failures to Connect to SkyX
    '''
    def __init__(self, value):
        ''' init'''
        super(SkyxConnectionError, self).__init__(value)
        self.value = value

    def __str__(self):
        ''' returns the error string '''
        return repr(self.value)

class SkyxTypeError(Exception):
    ''' Exception for Failures to Connect to SkyX
    '''
    def __init__(self, value):
        ''' init'''
        super(SkyxTypeError, self).__init__(value)
        self.value = value

    def __str__(self):
        ''' returns the error string '''
        return repr(self.value)
    
@Singleton
class SkyXConnection(object):
    ''' Class to handle connections to TheSkyX
    '''
    def __init__(self, host="localhost", port=3040):
        ''' define host and port for TheSkyX.
        '''
        self.host = host
        self.port = port
        
    def reconfigure(self,host="localhost", port=3040):
        ''' If we need to chane ip we can do so this way'''
        self.host = host
        self.port = port
                
    def _send(self, command):
        ''' sends a js script to TheSkyX and returns the output.
        '''
        try:
            logger.debug(command)
            sockobj = socket(AF_INET, SOCK_STREAM)
            sockobj.connect((self.host, self.port))
            sockobj.send(bytes('/* Java Script */\n' +
                               '/* Socket Start Packet */\n' + command +
                               '\n/* Socket End Packet */\n', 'utf8'))
            oput = sockobj.recv(2048)
            print(command)
            oput = oput.decode('utf-8')
            logger.debug(oput)
            sockobj.shutdown(SHUT_RDWR)
            sockobj.close()
            print("here")
            return oput.split("|")[0]
        except error as msg:
            raise SkyxConnectionError("Connection to " + self.host + ":" + \
                                      str(self.port) + " failed. :" + str(msg))

    def find(self, target):
        ''' Find a target
            target can be a defined object or a decimal ra,dec
        '''
        output = self._send('sky6StarChart.Find("' + target + '")')
        if output == "undefined":
            return True
        else:
            raise SkyxObjectNotFoundError(target)
                                    
 




