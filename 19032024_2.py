import hashlib
import json
import datetime
import random

# Класс, представляющий отдельный блок в блокчейне
class Block:
    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index  # Номер блока в цепочке
        self.timestamp = timestamp  # Временная метка создания блока
        self.data = data  # Данные блока (в данном случае, данные о голосовании)
        self.previous_hash = previous_hash  # Хэш предыдущего блока
        self.hash = self.calculate_hash()  # Хэш текущего блока

    # Метод для вычисления хэша блока
    def calculate_hash(self):
        block_data = str(self.index) + str(self.timestamp) + str(self.data) + str(self.previous_hash)
        return hashlib.sha256(block_data.encode()).hexdigest()

# Класс, представляющий блокчейн
class Blockchain:
    def __init__(self):
        # Создание блока-генезиса (первого блока в цепочке)
        self.chain = [self.create_genesis_block()]

    # Метод для создания блока-генезиса
    def create_genesis_block(self):
        return Block(0, datetime.datetime.now(), "Genesis Block", "0")

    # Метод для получения последнего блока в цепочке
    def get_latest_block(self):
        return self.chain[-1]

    # Метод для добавления нового блока в цепочку
    def add_block(self, new_block):
        new_block.previous_hash = self.get_latest_block().hash  # Установка хэша предыдущего блока в новом блоке
        new_block.hash = new_block.calculate_hash()  # Вычисление хэша нового блока
        self.chain.append(new_block)  # Добавление нового блока в цепочку

    # Метод для проверки валидности цепочки блоков
    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            # Проверка хэша текущего блока
            if current_block.hash != current_block.calculate_hash():
                return False

            # Проверка связности блоков по хэшам
            if current_block.previous_hash != previous_block.hash:
                return False

        return True

# Генерация данных о голосовании
def generate_votes(num_voters, candidates):
    votes = []
    for i in range(num_voters):
        voter_id = i + 1
        vote = {"voter_id": voter_id, "candidate": random.choice(candidates)}
        votes.append(vote)
    return votes

# Пример использования блокчейн-сети
my_blockchain = Blockchain()

# Генерация данных о голосовании
num_voters = 100
candidates = ["Путин", "Даванков", "Слуцкий", "Харитонов"]
votes = generate_votes(num_voters, candidates)

# Добавление блока с данными о голосовании
for index, vote in enumerate(votes):
    my_blockchain.add_block(Block(index + 1, datetime.datetime.now(), vote, my_blockchain.get_latest_block().hash))

# Проверка валидности цепочки блоков до изменения
print("\nIs blockchain valid ?", my_blockchain.is_chain_valid())

# Вывод результатов голосования до попытки изменения
print("\nРезультаты голосования до попытки изменения:")
results_before = {}
for block in my_blockchain.chain:
    if block.index != 0:  # Пропускаем блок-генезис
        candidate = block.data["candidate"]
        if candidate in results_before:
            results_before[candidate] += 1
        else:
            results_before[candidate] = 1

for candidate, votes in results_before.items():
    print(f"{candidate}: {votes} голосов")

# Попытка изменить голос в блоке цепочки (для тестирования)
block_to_tamper = my_blockchain.chain[42]
print("\nПопытка изменить голос в блоке цепочки...")
print("Данные о голосовании до изменения:", block_to_tamper.data)

# Изменение данных о голосовании
block_to_tamper.data["candidate"] = "Лукашенко"

# Пересчет хэшей для поддержания валидности цепочки
for i in range(42, len(my_blockchain.chain)):
    current_block = my_blockchain.chain[i]
    current_block.hash = current_block.calculate_hash()
    if i < len(my_blockchain.chain) - 1:
        my_blockchain.chain[i + 1].previous_hash = current_block.hash

# Проверка валидности цепочки блоков после попытки изменения
print("\nIs blockchain valid ?", my_blockchain.is_chain_valid())

# Вывод результатов голосования после попытки изменения
print("\nРезультаты голосования после попытки изменения:")
results_after_tampering = {}
for block in my_blockchain.chain:
    if block.index != 0:  # Пропускаем блок-генезис
        candidate = block.data["candidate"]
        if candidate in results_after_tampering:
            results_after_tampering[candidate] += 1
        else:
            results_after_tampering[candidate] = 1

for candidate, votes in results_after_tampering.items():
    print(f"{candidate}: {votes} голосов")

