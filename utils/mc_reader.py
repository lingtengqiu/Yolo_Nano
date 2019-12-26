__all__ = ['MemcachedReader']

import sys
sys.path.append('/mnt/lustre/share/pymc/py3')
# import mc


class MemcachedReader:

    def __init__(self,
                 server_list_conf="/mnt/lustre/share/memcached_client/server_list.conf",
                 client_conf="/mnt/lustre/share/memcached_client/client.conf"):
        self.server_list_conf = server_list_conf
        self.client_conf = client_conf

    def __call__(self, path):
        import mc
        mclient = mc.MemcachedClient.GetInstance(self.server_list_conf, self.client_conf)
        value = mc.pyvector()
        mclient.Get(path, value)
        content = mc.ConvertBuffer(value)
        return bytes(content)