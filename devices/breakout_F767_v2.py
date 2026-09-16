from devices.nucleo_connector import Port, Motor, Photometry_port, ESP_UART_port



class Breakout_F767_v2:
    
    def __init__(self):
        self.port_1 = Port(DIO_A="PD5", DIO_B="PD6", POW_A="PD11", POW_B="PD13", POW_C="PD12", UART=2)
        self.port_2 = Port(DIO_A="PG2",DIO_B="PG3", POW_A="PB10",POW_B="PA6",POW_C="PB6")
        self.port_4 = Port(DIO_A="PB8", DIO_B="PB9", DIO_C="PA5", POW_A="PB2", POW_B="PC4", DAC=2, I2C=1)
        self.port_5 = Port(DIO_A="PC6", DIO_B="PC7", DIO_C="PB1", POW_A="PA2", POW_B="PE8")
        self.port_6 = Port(DIO_A="PF5", DIO_B="PA3", DIO_C="PF4", POW_A="PE7", POW_B="PD14")
        self.port_7 = Port(DIO_A="PF10", DIO_B="PE12", DIO_C="PD15", POW_A="PE10", POW_B="PE9")
        self.port_8 = Port(DIO_A="PF14", DIO_B="PF15", DIO_C="PA4", POW_A="PE14", POW_B="PE15", DAC=1, I2C=4)
        self.port_9 = Port(DIO_A="PF12", DIO_B="PE0", DIO_C="PC2", POW_A="PE13", POW_B="PE11")
        self.port_10 = Port(DIO_A="PG11", DIO_B="PG6", POW_A="PF13", POW_B="PB13", POW_C="PF3")

        self.motor1 = Motor(DIR="PG9", STEP="PG13", EN="PD9")
        self.motor2 = Motor(DIR="PE6", STEP="PE1", EN="PF8")
        self.motor3 = Motor(DIR="PG1", STEP="PD0", EN="PG0")
        self.motor4 = Motor(DIR="PF2", STEP="PF9", EN="PD1")
        self.motor5 = Motor(DIR="PE3", STEP="PE4", EN="PE5")
        self.motor6 = Motor(DIR="PC12", STEP="PD7", EN="PE2")
        self.motor7 = Motor(DIR="PC11", STEP="PC10", EN="PD2")
        self.motor8 = Motor(DIR="PD4", STEP="PD3", EN="PC3")

        # pyPhotometry acquisition board connection.
        self.photometry1 = Photometry_port(
            DIGITAL1="PC0", DIGITAL2="PA0",
            SIGNAL1="PA1", SIGNAL2="PC1",
            LED1CON="PA4", LED2CON="PA5" )

        # ESP32-S3 UART mic interface.
        self.esp_uart1 = ESP_UART_port(TX="PF7", RX="PF6",
            GPIO1="PA13", GPIO2="PA14", UART=7)



