import logging
import os
from typing import Dict, Any, Optional
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
import requests

logger = logging.getLogger(__name__)

class TwilioWhatsAppClient:
    def __init__(self):
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")
        self.whatsapp_number = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
        
        if self.account_sid and self.auth_token:
            self.client = Client(self.account_sid, self.auth_token)
        else:
            self.client = None
            logger.warning("Twilio credentials not set. Responses won't actually send.")

    def parse_incoming_message(self, form_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parses the Twilio POST request data."""
        return {
            "from": form_data.get("From", ""),
            "body": form_data.get("Body", ""),
            "media_url": form_data.get("MediaUrl0", ""),
            "media_type": form_data.get("MediaContentType0", "")
        }

    def download_media(self, media_url: str, save_path: str) -> Optional[str]:
        """Downloads an image or audio file sent via WhatsApp."""
        if not media_url:
            return None
            
        try:
            # Twilio media requires HTTP Basic Auth using SID/Token
            response = requests.get(media_url, auth=(self.account_sid, self.auth_token))
            if response.status_code == 200:
                with open(save_path, "wb") as f:
                    f.write(response.content)
                return save_path
            else:
                logger.error(f"Failed to download Twilio media. Status: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Twilio media download error: {e}")
            return None

    def generate_twiml_response(self, text_message: str, media_url: Optional[str] = None) -> str:
        """Generates TwiML XML to respond to the webhook instantly."""
        resp = MessagingResponse()
        msg = resp.message()
        msg.body(text_message)
        if media_url:
            msg.media(media_url)
        return str(resp)

    def send_proactive_message(self, to_number: str, message: str, media_url: Optional[str] = None):
        """Sends an async proactive message to a WhatsApp number. Slices message if > 1600 chars."""
        if not self.client:
            logger.info(f"Mock Send to {to_number}: {message}")
            return
            
        try:
            # Twilio WhatsApp limit is 1600 chars. Slice it safely.
            chunk_size = 1550
            chunks = [message[i:i + chunk_size] for i in range(0, len(message), chunk_size)]
            
            for index, chunk in enumerate(chunks):
                kwargs = {
                    "body": chunk,
                    "from_": self.whatsapp_number,
                    "to": to_number
                }
                # Only attach media to the last chunk
                if media_url and index == len(chunks) - 1:
                    kwargs["media_url"] = [media_url]
                    
                self.client.messages.create(**kwargs)
                logger.info(f"Sent chunk {index+1}/{len(chunks)} to {to_number}")
        except Exception as e:
            logger.error(f"Failed to send Twilio message: {e}")

twilio_client = TwilioWhatsAppClient()
