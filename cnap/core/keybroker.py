"""A Keybroker module.

This module provides an object-oriented design for key broker client to connect with
key broker server (KBS), get model decryption key from the KBS.

Classes:
    KeyBrokerClientBase: An abstract base class for key broker client.
    AmberKeyBrokerClient: A concrete class implementing the KeyBrokerClientBase
      for Amber KBS.
"""

import base64
import logging
import requests
import struct

from abc import ABC, abstractmethod
from ccnp import Quote
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

LOG = logging.getLogger(__name__)

class KeyBrokerClientBase(ABC):
    """An abstract base class for key broker client.

    This class serves as a blueprint for subclasses that need to implement
    `get_key` methods for different types of key broker server.
    """

    @abstractmethod
    def get_key(self, server_url: str, key_id: str):
        """Get a key from key broker server.

        This method is used to get a key from key broker server.

        Args:
            server_url (str): The key broker server url.
            key_id (str): The id of the key.

        Raises:
            ValueError: If the server_url or key_id is None.
            RuntimeError: If get quote or get key failed.
            NotImplementedError: If the subclasses don't implement the method.
        """
        raise NotImplementedError("Subclasses should implement connect() method.")


class AmberKeyBrokerClient(KeyBrokerClientBase):
    """Amber implementation for key broker client.

    This class implement `connect`, `publish_frame` methods defined in
    `KeyBrokerClientBase` abstract base class for Amber stream broker server.

    """
    def get_key(self, server_url: str, key_id: str) -> bytes:
        """ Get model key by key ID from Amber KBS.

        The flow to get a key:
        * generate 2048 bit RSA key pair
        * encode public key to base64 as the user data in request body
        * get quote with the user data
        * request encrypted key and SWK by the quote and user data
        * decrypt key by SWK
  
        Amber KBS key transfer API: v1/keys/keyid/transfer
        request headers:
            Accept:application/json
            Content-Type:application/json
            Attestation-Type:TDX
        request body:
            {
            "quote":"",
            "user_data":""
            }
        response body:
            {
            "wrapped_key":"",
            "wrapped_swk":""
            }

        Args:
            server_url (str): The key broker server url.
            key_id (str): The id of the key.

        Raises:
            ValueError: If the server_url or key_id is None.
            RuntimeError: If get quote or get key failed.
            NotImplementedError: If the subclasses don't implement the method.
        """
        if server_url is None:
            raise ValueError("KBS server url can not be None")
        if key_id is None:
            raise ValueError("KBS key id can not be None")

        private_key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
        public_key = private_key.public_key()
        pubkey_der = public_key.public_bytes(encoding=serialization.Encoding.DER, format=serialization.PublicFormat.SubjectPublicKeyInfo)
        LOG.debug("Getting TDX Quote by CCNP.")

        user_data = base64.b64encode(pubkey_der).decode('utf-8')
        tdquote = Quote.get_quote(nonce=user_data)
        if tdquote is None:
            raise RuntimeError("Get TDX Quote failed")
        quote = base64.b64encode(tdquote.quote).decode('utf-8')

        req = {
            "quote": quote,
            "user_data": user_data
        }
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Attestation-Type": "TDX"
        }
        amber_api = f"{server_url}/keys/{key_id}/transfer"
        LOG.debug("Getting key from Amber KBS.")
        # retry 3 times for Amber KBS server issue
        resp = None
        for _ in range(3):
            try:
              resp = requests.post(amber_api, json=req, headers=headers, verify=False)
              if resp.status_code in [200]:
                  break
            except requests.exceptions.ConnectionError:
                pass
        if resp is None or resp.status_code != 200:
            raise RuntimeError("Unexpected response from Amber KBS")

        resp_body = resp.json()
        if "wrapped_key" not in resp_body or "wrapped_swk" not in resp_body:
            raise RuntimeError("Empty key response from Amber KBS")

        wrapped_key = base64.b64decode(resp_body['wrapped_key'])
        wrapped_swk = base64.b64decode(resp_body['wrapped_swk'])
        LOG.debug("Decrypting the SWK.")
        swk = private_key.decrypt(
          wrapped_swk,
          padding.OAEP(
              mgf=padding.MGF1(algorithm=hashes.SHA256()),
              algorithm=hashes.SHA256(),
              label=None
          )
        )

        key = self.decrypt_data(wrapped_key, swk)
        return key

    def decrypt_data(self, encrypted_data, key) -> bytes:
        """Decrypt model by a given key

        Normally, the encrypted data format should be:
         -------------------------------------------------------------------
        | 12 bytes header | [12] bytes IV | encrypted data | [16] bytes tag |
         -------------------------------------------------------------------
        and the 12 bytes header:
         -----------------------------------------------------------
        | uint32 IV length | uint32 tag length | uint32 data length |
         -----------------------------------------------------------

        Args:
            encrypted_data (bytes): The encrypted data for decryption.
            key (bytes): The key for decryption.

        Raises:
            ValueError: If the encrypted_data or key is None.
        """
        if encrypted_data is None:
            raise ValueError("The encrypted data can not be None")
        if key is None:
            raise ValueError("The key can not be None")

        header_len = 12
        iv_len, tag_len, data_len = struct.unpack('<3I', encrypted_data[:header_len])
        iv = encrypted_data[header_len : (iv_len + header_len)]
        data = encrypted_data[(iv_len + header_len) : -tag_len]
        tag = encrypted_data[-tag_len:]

        LOG.debug("Decrypt data, IV len %d, tag len %d, data len %d", iv_len, tag_len, data_len)
        decryptor = Cipher(algorithms.AES(key), modes.GCM(iv, tag)).decryptor()
        decrypted_data = decryptor.update(data) + decryptor.finalize()
        return decrypted_data
