import random  # Импортируем модуль random для случайного выбора кандидата
from hashlib import \
    sha256  # Импортируем функцию sha256 из модуля hashlib для вычисления хэшей блоков
import json  # Импортируем модуль json для преобразования данных в формат JSON
import time  # Импортируем модуль time для работы с временными метками


class Block:
    def __init__(self, index, timestamp, data, previous_hash):
        """
        Конструктор класса Block.

        :param index: Номер блока в цепи.
        :param timestamp: Временная метка создания блока.
        :param data: Данные блока (в нашем случае информация о голосе).
        :param previous_hash: Хэш предыдущего блока в цепи.
        """
        self.index = index  # Номер блока в цепи
        self.timestamp = timestamp  # Временная метка создания блока
        self.data = data  # Данные блока (в нашем случае информация о голосе)
        self.previous_hash = previous_hash  # Хэш предыдущего блока в цепи
        self.hash = self.calculate_hash()  # Хэш текущего блока

    def calculate_hash(self):
        """
        Метод для расчета хэша блока.
        :return: Хэш текущего блока.
        """
        return sha256((str(self.index) + str(self.timestamp) + json.dumps(
            self.data) + self.previous_hash).encode()).hexdigest()


class Blockchain:
    def __init__(self):
        """
        Конструктор класса Blockchain.
        """
        self.chain = [
            self.create_genesis_block()]  # Инициализация цепи с генезис-блоком

    def create_genesis_block(self):
        """
        Метод для создания генезис-блока.
        :return: Генезис-блок.
        """
        return Block(0, time.time(), "Genesis Block", "0")

    def get_latest_block(self):
        """
        Метод для получения последнего блока в цепи.
        :return: Последний блок в цепи.
        """
        return self.chain[-1]

    def add_block(self, new_block):
        """
        Метод для добавления нового блока в цепь.
        :param new_block: Новый блок для добавления.
        """
        new_block.previous_hash = self.get_latest_block().hash
        new_block.hash = new_block.calculate_hash()
        self.chain.append(new_block)

    def is_chain_valid(self):
        """
        Метод для проверки целостности цепи блоков.
        :return: True, если цепь валидна, иначе False.
        """
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            # Проверка хэша текущего блока
            if current_block.hash != current_block.calculate_hash():
                return False
            # Проверка связи между текущим и предыдущим блоками
            if current_block.previous_hash != previous_block.hash:
                return False
        return True


class Vote:
    def __init__(self, candidate):
        """
        Конструктор класса Vote.
        :param candidate: Выбранный кандидат.
        """
        self.candidate = candidate  # Выбранный кандидат
        self.voter_public_key = None  # Публичный ключ избирателя (для примера)
        self.signature = None  # Подпись голоса (для примера)

    def sign_vote(self, voter_public_key):
        """
        Метод для подписи голоса (для примера).
        :param voter_public_key: Публичный ключ избирателя.
        """
        self.voter_public_key = voter_public_key
        self.signature = "Digital Signature"  # Здесь должна быть реальная криптографическая подпись


# Пример использования
blockchain = Blockchain()  # Создаем экземпляр блокчейна

candidates = ["Путин", "Харитонов", "Даванков", "Слуцкий"]  # Список кандидатов

# Создаем голоса для каждого кандидата и подписываем их
for _ in range(100):  # Имитируем голосование на 100 человек
    candidate = random.choice(candidates)  # Случайный выбор кандидата
    vote = Vote(candidate)
    vote.sign_vote("Public Key Voter")  # Подписываем голос (для примера)
    blockchain.add_block(Block(len(blockchain.chain), time.time(),
                               {"candidate": vote.candidate},
                               blockchain.get_latest_block().hash))

# Проверяем целостность блокчейна
print("Is blockchain valid?", blockchain.is_chain_valid())

# Выводим результаты голосования
print("Результаты голосования:")
results = {candidate: 0 for candidate in
           candidates}  # Создаем словарь для подсчета голосов
for block in blockchain.chain[1:]:  # Пропускаем генезис-блок
    candidate = block.data[
        "candidate"]  # Получаем имя кандидата из данных блока
    results[
        candidate] += 1  # Увеличиваем количество голосов для соответствующего кандидата

# Выводим результаты голосования
for candidate, votes_count in results.items():
    print(f"{candidate}: {votes_count} голосов")

# Попытка изменить результаты голосования
print("\nПопытка изменить результаты голосования:")

# Выбираем случайный блок для изменения голоса
block_to_modify = random.choice(blockchain.chain[1:])
print("Выбранный блок для изменения:", block_to_modify.hash)

# Выбираем случайного кандидата для изменения голоса
new_candidate = random.choice(candidates)
print("Новый кандидат:", new_candidate)

# Меняем голос в выбранном блоке
block_to_modify.data["candidate"] = new_candidate
block_to_modify.hash = block_to_modify.calculate_hash()  # Пересчитываем хэш блока

# Проверяем целостность блокчейна после изменения голоса
print("Is blockchain valid?",
      blockchain.is_chain_valid())

# Повторно выводим результаты голосования после изменения голоса
print("\nРезультаты голосования после попытки изменения:")
modified_results = {candidate: 0 for candidate in
                    candidates}  # Создаем словарь для подсчета голосов после изменения
for block in blockchain.chain[1:]:  # Пропускаем генезис-блок
    candidate = block.data[
        "candidate"]  # Получаем имя кандидата из данных блока
    modified_results[
        candidate] += 1  # Увеличиваем количество голосов для соответствующего кандидата после изменения

# Выводим результаты голосования после изменения
for candidate, votes_count in modified_results.items():
    print(f"{candidate}: {votes_count} голосов")
