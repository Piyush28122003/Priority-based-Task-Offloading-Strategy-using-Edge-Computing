class Server:
    def __init__(self, id, x, y, ram, rom):
        self.id = id
        self.x = x
        self.y = y
        self.ram = ram
        self.rom = rom
        self.available_ram = ram
        self.available_rom = rom

    def can_handle_task(self, task):
        return self.available_ram >= task.ram and self.available_rom >= task.rom
