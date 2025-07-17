import pika


class Queue:
    def __init__(self, name: str, host: str = "localhost") -> None:
        self._name = name
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))
        self.channel = connection.channel()
        self.channel.queue_declare(queue=name)

    def push(self, data) -> None:
        self.channel.basic_publish(exchange="", routing_key=self._name, body=data)

    def pop(self) -> bytes | None:
        method_frame, _, body = self.channel.basic_get(queue=self._name)
        if method_frame:
            self.channel.basic_ack(method_frame.delivery_tag)
            return body
        return None

