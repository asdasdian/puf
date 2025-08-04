/*
 * Copyright (c) 2013 - 2015, Freescale Semiconductor, Inc.
 * Copyright 2016-2017 NXP
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <stdio.h>
#include <string.h>
#include "fsl_device_registers.h"
#include "fsl_debug_console.h"
#include "board.h"

#include "pin_mux.h"
#include "clock_config.h"
#include "fsl_puf.h"
#include "fsl_hashcrypt.h"
#include "fsl_iap.h"
#include "fsl_iap_ffr.h"
#include "sensor_puf_app.h"
/*******************************************************************************
 * Definitions
 ******************************************************************************/
#define CORE_CLK_FREQ CLOCK_GetFreq(kCLOCK_CoreSysClk)
#define PUF_DISCHARGE_TIME 400

#define NUM_OF_KEYS 2


/*******************************************************************************
 * Prototypes
 ******************************************************************************/
typedef uint8_t keyCode_t[PUF_GET_KEY_CODE_SIZE_FOR_KEY_SIZE(kPUF_KeySizeMax)];

typedef struct
{
    char ** menutext;
    void (**menufnc)(void);
}sMenu;

typedef struct
{
  uint8_t activationCode[PUF_ACTIVATION_CODE_SIZE]; 
  keyCode_t keyCode[NUM_OF_KEYS];
}sPufData;


/*******************************************************************************
 * Function definition
 ******************************************************************************/
/*!
 * @brief Main function
 */
void MenuFindFnc(sMenu * menuList, uint32_t menuListLen, char ** actualmenu, void (*** fnc)(void));
uint32_t MenuLength(char ** menu);
void MenuPrint( char ** menu);
void PufAllowPrint(PUF_Type *base);
void PufStatPrint(PUF_Type *base);
uint32_t IsEmptyMem(uint8_t * adr, uint32_t len);
void PrintKeyCode(uint8_t * kc, uint32_t size, uint32_t segmentation);
void StoreKeyCode(keyCode_t keycode, uint32_t keycodesize);
void LoadKeyCode(uint8_t ** keycode);
uint32_t KeyCodeCheck(uint8_t * keycode, uint32_t * keytype, uint32_t * keyidx, uint32_t * keysize);
void PrintMem(uint8_t * adr, uint32_t len, uint32_t seg);
uint32_t Flash_StoreAC(uint8_t *acBuf);
void Flash_ReadAC(uint8_t ** acBuf);
void  Flash_ReadKC(uint8_t **kcBuf, uint32_t keyIdx);
uint32_t Flash_StoreKC(uint8_t *kcBuf, uint32_t keyIdx, uint32_t size);
void verify_status(status_t status);

/*********************** Menu functions ***********************************/
void MainEnrollPuf(void);
void MainStartPuf(void);
void MainMiscPuf(void);
void MainGenerateKey(void);
void MainGetKey(void);
void MainAES(void);
void MainSensorPufDemo(void);
void MainBack(void);

void MiscStopPuf(void);
void MiscInitPuf(void);
void MiscZeroize(void);
void MiscBlockEnroll(void);
void MiscBlockSetKey(void);
void MiscBack(void);

void GetKey(void);
void KeyBack(void);

void EnrolPuf(void);
void StartPuf(void);
void StopPuf(void);
void InitPuf(void);
void Zeroize(void);
void BlockEnroll(void);
void BlockSetKey(void);
void GenerateUserKey(void);
void GenerateIntrinsicKey(void);
void TestAesEcb(void);

#define FLASHSTORE_BASEADR 0x80000
#define FLASHSTORE_LEN /*sizeof(sPufData)*/ 0x1000

/************************ PUF variables *********************************/
void (**actualfnc)(void);

sPufData pufData;
sPufData pufCmpaData;

status_t result;
char ** menu;

uint8_t pKeyCode[560];
flash_config_t flashInstance;

/*******************************************************************************
 * Code
 ******************************************************************************/

/************************** MENU *****************************************/
/************************ MAIN MENU **************************************/
char * mainmenu[] =
{
  "Enroll PUF",
  "Start and load AC to PUF",
  "Misc. PUF commands",
  "Generate Key Code",
  "Get Key from Key Code",
  "Encrypt / Decrypt AES block",
  "Sensor PUF Demo",
  "Back",
};

void (*mainmenufnc[])(void) =
{
  MainEnrollPuf,
  MainStartPuf,
  MainMiscPuf,
  MainGenerateKey,
  MainGetKey,
  MainAES,
  MainSensorPufDemo,
  MainBack,
};

/************************ MISC MENU **************************************/
char * miscmenu[] =
{
  "Init PUF ",
  "Stop PUF ",
  "Zeroize PUF ",
  "Disable Enroll PUF ",
  "Disable Key Generation",
  "Back",
};

void (*miscmenufnc[])(void) =
{
  MiscInitPuf,
  MiscStopPuf,
  MiscZeroize,
  MiscBlockEnroll,
  MiscBlockSetKey,
  MiscBack,
};

/************************ Key MENU **************************************/
char * setkeymenu[] =
{
  "Generate user key code",
  "Generate intrinsic key code",
  "Back",
};

void (*setkeymenufnc[])(void) =
{
  GenerateUserKey,
  GenerateIntrinsicKey,
  KeyBack,
};

/************************ MENU List **************************************/
sMenu menulist[] =
{
  {mainmenu, mainmenufnc},
  {miscmenu, miscmenufnc},
  {setkeymenu, setkeymenufnc},
};


/************************ PUF FUNCTIONALITY ***********************************/
/************************ ENROLL functions ************************************/
/******************************************************************************/
void EnrolPuf(void)
{
    uint32_t keystore;
    uint8_t ac[PUF_ACTIVATION_CODE_SIZE];
    memset(pufData.activationCode, 0, sizeof(pufData.activationCode));
    while (1)
    {
        /* Before any PUF use PUF peripheral has to be initialised*/
        result = PUF_Init(PUF, PUF_DISCHARGE_TIME, CORE_CLK_FREQ);
        if (result != kStatus_Success)
        {
            PRINTF("Error Initializing PUF!\r\n");
            break;
        }
        /* Perform enroll to get device specific PUF activation code */
        result = PUF_Enroll(PUF, ac, PUF_ACTIVATION_CODE_SIZE);
        if (result != kStatus_Success)
            PRINTF("Error during Enroll!\r\n");
        
        break;
    }
    
    if (result == kStatus_Success)
    {
        PRINTF("\r\nActivation Code (AC) was created!\r\n\n");
        PRINTF("Activation Code:");
        PrintMem(ac, PUF_ACTIVATION_CODE_SIZE, 16);   

        PRINTF("\r\nStore AC to\r\n1. RAM activationCode\r\n2. FLASH activationkeycode\r\n");
        while(1)
        {
          SCANF("%d", &keystore);
          keystore--;
          if(keystore < 1)
          {
              memcpy(pufData.activationCode, ac, PUF_ACTIVATION_CODE_SIZE);
              break;
          }
          else if(keystore < 2)
          {
        	  Flash_StoreAC(ac);
        	  break;
          }
          else
              PRINTF("\r\nInput value %d is bad\r\n", ++keystore);
        }
    }
    else
    {
        PRINTF("Activation Code (AC) wasn't created!\r\n");
    }
}


/************************ START & load AC  ************************************/
/******************************************************************************/
void StartPuf(void)
{
  uint32_t acidx;
  uint8_t * pac;
  status_t status;
  while(1)
  {
      PRINTF("\nChoose AC source\r\n1. RAM\r\n2. Flash\r\n3. CMPA key store\r\n");
      SCANF("%d", &acidx);
      acidx--;
      if(acidx > 2)
      {
          PRINTF("\nBad value\r\n"); continue;
      }
      
      if(acidx == 0)
        pac = pufData.activationCode;
      else if (acidx == 1)
      {
          Flash_ReadAC(&pac);
      }
      else if (acidx == 2)
      {
        memset((void *)&pufCmpaData.activationCode, 0, sizeof(pufCmpaData.activationCode));
        status = FFR_KeystoreGetAC(&flashInstance, (uint8_t *)&pufCmpaData.activationCode);
        if (status != kStatus_Success)
        {
          PRINTF("\r\nRead AC from CMPS failed\r\n");
          break;
        }
        pac = (uint8_t *)pufCmpaData.activationCode;
      }        
      
      PRINTF("Activation Code:");
      PrintMem(pac, sizeof(pufData.activationCode), 16);   
      
      PUF_Deinit(PUF, PUF_DISCHARGE_TIME, CORE_CLK_FREQ);  
      
      /* Reinitialize PUF after enroll */
      status = PUF_Init(PUF, PUF_DISCHARGE_TIME, CORE_CLK_FREQ);
      if (status != kStatus_Success)
      {
          PRINTF("\nError Initializing PUF!\r\n");
          break;
      }

      /* Start PUF by loading generated activation code */
      status = PUF_Start(PUF, pac, sizeof(pufData.activationCode));
      if (status != kStatus_Success)
      {
          PRINTF("\nError during Start !\r\n");
          break;
      }
      PRINTF("\nThe PUF is started\r\n");
      break;
  }
}


/************************ Misc functions **************************************/
void InitPuf(void)
{
  status_t status;
  status = PUF_Init(PUF, PUF_DISCHARGE_TIME, CORE_CLK_FREQ);
  if(status == kStatus_Success)
    PRINTF("\n\nPUF Init succesfull\r\n");
  else
    PRINTF("\n\nPUF Init failed\r\n");
}

void BlockSetKey(void)
{
  PUF_BlockSetKey(PUF);
}

void BlockEnroll(void)
{
    PUF_BlockEnroll(PUF);
}

void Zeroize(void)
{
  status_t status;
  status = PUF_Zeroize(PUF);
  if(status == kStatus_Success)
    PRINTF("\n\nZeroize succesfull\r\n");
  else
    PRINTF("\n\nZeroize failed\r\n");
}

void StopPuf(void)
{
  PUF_Deinit(PUF, PUF_DISCHARGE_TIME, CORE_CLK_FREQ);
}





/************************ User Key functions **************************************/
void GenerateUserKey(void)      
{
    uint32_t kidx, keylen, keylenbyte, keylenbit, keycodesize;
    uint8_t key[512];
    keyCode_t tempKC;
    
    PRINTF("\nEnter key index 0..15\r\nIn order to send key to AES or PRINCE use key index 0\r\n");
    SCANF("%d", &kidx);
    if(kidx > 15)
    {
      PRINTF("\nBad key index, should be 0..15 \r\n");
      return;
    }    
    PRINTF("\n\nEnter key size 1..64 (64..4096)-bits. \r\nAES allowed key size: 128/196/256-bit \r\nPRINCE allowed key size 128-bit\r\n");
    SCANF("%d", &keylen);
    
    keylen &= 0x3F;  
    if((kidx > 63))
    {
      PRINTF("\nBad key index, should be 1..64 \r\n");
      return;
    }
    keylenbyte = (keylen == 0 ? 512 : 8*keylen);
    keylenbit = keylenbyte*8;
    keycodesize = PUF_GET_KEY_CODE_SIZE_FOR_KEY_SIZE(keylenbyte);
    PRINTF("\nEnter your user password %dB long \r\nshorter will be padded by SPACEs \r\nlonger cuted\r\n", keylenbyte);
    memset(key,' ',keylenbyte);
    SCANF("%s", key);
    uint8_t se = strlen((char const *)key);
    key[se] = ' ';
    
    PRINTF("\n\r\nGenerating user Key Code (KC) with Index %d, and Key length %d-bits\r\n", kidx, keylenbit);
    PRINTF("\nKey:");
    PrintMem(key, keylenbyte, 16);
     
    result = PUF_SetUserKey(PUF, (puf_key_index_register_t)kidx, key, keylenbyte, /*pufData.keyCode[0]*/ tempKC, keycodesize );
    if (result != kStatus_Success)
    {
        PRINTF("\r\n!!!!! Error setting user key! Is the PUF already started.\r\n");
    }
    else
    {
        PRINTF("\r\nKey Code (KC) is generated succesfully\r\n");
        PrintKeyCode(tempKC, keycodesize ,16);
        StoreKeyCode(tempKC, keycodesize);       
     }
}

/************************ Intrinsic Key FUNCTIONS ******************************/
void GenerateIntrinsicKey(void)
{
    uint32_t kidx, keylen, keylenbyte, keylenbit, keycodesize;
    keyCode_t tempKC;

    PRINTF("\nEnter key index 0..15\r\nIn order to send key to AES or PRINCE use key index 0\r\n");
    SCANF("%d", &kidx);
    if(kidx > 15)
    {
      PRINTF("\nBad key index, should be 0..15 \r\n");
      return;
    }    
    PRINTF("\n\nEnter key size 1..64 (64..4096)-bits. \r\nAES allowed key size: 128/196/256-bit \r\nPRINCE allowed key size 128-bit\r\n");
    SCANF("%d", &keylen);
    keylen &= 0x3F;  
    if((kidx > 63))
    {
      PRINTF("\nBad key index, should be 1..64 \r\n");
      return;
    }
    keylenbyte = (keylen == 0 ? 512 : 8*keylen);
    keylenbit = keylenbyte*8;
    keycodesize = PUF_GET_KEY_CODE_SIZE_FOR_KEY_SIZE(keylenbyte);

    PRINTF("\r\nGenerating Intrinsic Key Code (KC) with Index %d, and Key length %d-bits\r\n", kidx, keylenbit);
    
    result = PUF_SetIntrinsicKey(PUF, (puf_key_index_register_t)kidx, keylenbyte, tempKC, keycodesize );
    if (result != kStatus_Success)
    {
      PRINTF("!!!!! Error setting user key! Is the PUF already started.\r\n");
    }
    else
    {
        PRINTF("Key Code (KC) is generated succesfully\r\n");
        PrintKeyCode(tempKC, keycodesize, 16);
                StoreKeyCode(tempKC, keycodesize);       
    }
}


/************************ Intrinsic Key FUNCTIONS ******************************/
void GetKey(void)
{
    uint32_t keyidx, keyslot, keycodesize, keysize, keytype, status;
    uint8_t * keycode;
    
    uint8_t key[512];
    
LoadKeyCode(&keycode);
      if(keycode == NULL)
      {
        PRINTF("\r\nError loading code");
        return;
      }
      
      status = KeyCodeCheck(keycode, &keytype, &keyidx, &keysize);
      if(status != kStatus_Success)
      {
        keysize = 64; // limit size to max key size in order to size is non sence
        PRINTF("\r\nError in Key Code header");
      }
                   
      keycodesize = PUF_GET_KEY_CODE_SIZE_FOR_KEY_SIZE(keysize);
      PrintKeyCode(keycode, keycodesize, 16);
      
      // print out info about key code
      PRINTF("\r\nReconstruction of keycode:\r\nsize %d bytes, index %d, type %s\r\n", keysize, keyidx, keytype == 0 ? "user":"intrinsic");
      
      if(keyidx > 0) // print to terminal
      {
          PRINTF("\r\nKey Code index > 0, Key will be printed");
          result = PUF_GetKey(PUF, keycode, keycodesize, key, keysize);
          if (result != kStatus_Success)
          {
              PRINTF("\r\nError reconstructing user key!\r\n");
          }
          
          else
          {
              PRINTF("\r\nKey:\r\n");
              PrintMem(key, keysize, 16 );
          }
      }
      else // key code index == 0 -> Key will be sent to internal hw. bus
      {
        PRINTF("\r\nKey Code index = 0, Key will sent to internal hw. bus");
        while(1)
        { 
          PRINTF("\r\nInput keyslot value <0..3>\r\nKey will be send to \r\n1: AES\r\n2: Prince 1\r\n3: Prince 2\r\n4: Prince 3\r\n");
          SCANF("%d", &keyslot);
          keyslot--;
          if(keyslot < 4)
              break;
          
          PRINTF("\r\nBad value, enter again\r\n"); 
        }
          result = PUF_GetHwKey(PUF, keycode, keycodesize, (puf_key_slot_t)keyslot, rand());
          if (result != kStatus_Success)
          {
              PRINTF("\r\nError reconstructing key to HW bus!\r\n");
          }
          else
          {
              PRINTF("\r\nKey was sent to HW bus!\r\n");
          }
    }
}


/****************************** AES functions **********************************/
void TestAesEcb(void)
{
    uint8_t plaintext[64];
    uint8_t keyAes[32];
    uint8_t cipher[16];
    uint8_t output[16];
    status_t status;
    hashcrypt_handle_t m_handle;
    uint8_t * keycode;
    uint32_t keysize, keycodesize, keyidx, keytype;

    memset(plaintext, ' ', sizeof(plaintext));
    PRINTF("\r\nInput plain text 16B long, shorter will be padded with spaces 0x20\r\n");
    SCANF("%s", &plaintext);
    plaintext[strlen((char const *)plaintext)] = ' ';
    

    HASHCRYPT_Init(HASHCRYPT);

     PRINTF("\r\nAES key:\r\n1. Secret key\r\n2. User key\r\n");
     uint32_t keysource;
     SCANF("%d", &keysource);
     
     if(keysource == 1) 
     {
       m_handle.keyType = kHASHCRYPT_SecretKey;
       PRINTF("\r\nSecret Key. Will use Key already sent to AES\r\n");
     }
     else
     {
        m_handle.keyType = kHASHCRYPT_UserKey;
        PRINTF("\r\nChoose key code to be used by AES\r\n");
    
        LoadKeyCode(&keycode);
            
        if(keycode == NULL)
        {
          PRINTF("\r\nError loading code");
          return;
        }
        
        status = KeyCodeCheck(keycode, &keytype, &keyidx, &keysize);
        if (status != kStatus_Success)
        {
            keysize = 64; // limit size to max key size in order to size is non sence
            PRINTF("\r\nError key code check\r\n");
        }
         keycodesize = PUF_GET_KEY_CODE_SIZE_FOR_KEY_SIZE(keysize);
         PRINTF("\r\nReconstruction of keycodex:\r\nsize %d bytes, index %d, type %s\r\n", keysize, keyidx, keytype == 0 ? "user":"intrinsic");
    
        status = PUF_GetKey(PUF, keycode, keycodesize, keyAes, keysize);
        if (status != kStatus_Success)
        {
            PRINTF("\r\nError reconstructing user key!\r\n");
        }
        
        else
        {
            PRINTF("\r\nKey:\r\n");
            PrintMem(keyAes, keysize, 16 );
        }        
     }

    status = HASHCRYPT_AES_SetKey(HASHCRYPT, &m_handle, keyAes, 16);
    status = HASHCRYPT_AES_EncryptEcb(HASHCRYPT, &m_handle, plaintext, cipher, 16);

    PRINTF("\r\nPlain data:\r\n");
    PrintMem(plaintext, 16, 16);
    
    PRINTF("\r\nCipher:\r\n");
    PrintMem(cipher, sizeof(cipher), 16);

    status = HASHCRYPT_AES_DecryptEcb(HASHCRYPT, &m_handle, cipher, output, 16);

    PRINTF("\r\nDecrypted:\r\n");
    PrintMem(output, sizeof(output),16);
}


/********************************************************************/
/**************************** Main **********************************/
/********************************************************************/




int main(void)
{

    int32_t selection;
    uint32_t status;
    uint32_t failedAddress, failedData;
    uint8_t buf[0x1000] __attribute__ ((aligned (256)));

    memset(buf, 0, sizeof(buf));

    // switch off systick
     SysTick->CTRL = 0;

     // lets monitor busclock
     SYSCON->CLKOUTSEL = 0; // main clock
     SYSCON->CLKOUTDIV = 9; // divide by 10, reset = 0, halt = 0,

    /* Init board hardware. */
    /* attach FRO12MHz clock to FLEXCOMM0 (debug console) */
    CLOCK_AttachClk(BOARD_DEBUG_UART_CLK_ATTACH);
    BOARD_InitPins();
    BOARD_BootClockFROHF96M();
    BOARD_InitDebugConsole();

    memset(&flashInstance, 0, sizeof(flash_config_t));
    FLASH_Init(&flashInstance);
    FFR_Init(&flashInstance);

    PRINTF(" Asvin ID PUF \n");
    status = FLASH_Erase(&flashInstance, FLASHSTORE_BASEADR, sizeof(buf), kFLASH_ApiEraseKey);
    verify_status(status);
    PRINTF("Done!\r\n");

    /* Verify if the given flash range is successfully erased. */
    PRINTF("Calling FLASH_VerifyErase() API...\r\n");
    status = FLASH_VerifyErase(&flashInstance, FLASHSTORE_BASEADR, sizeof(buf));
    verify_status(status);
    if (status == kStatus_Success)
    {
        PRINTF("FLASH Verify Erase successful!\n");
    }

	status =  FLASH_Program(&flashInstance, FLASHSTORE_BASEADR, buf, sizeof(buf));
	verify_status(status);

	status = FLASH_VerifyProgram(&flashInstance, FLASHSTORE_BASEADR, sizeof(buf), buf,
		 &failedAddress, &failedData);
	verify_status(status);

    PRINTF("\r\n**************************************************\r\n");
    PRINTF(" The software coming with AN12324\r\n");
    PRINTF(" The PUF and AES application note example code v1.1\r\n");
    PRINTF(" Use this example to Enroll and start PUF and then\r\n");
    PRINTF(" generate KEYs and their KEY Code. Then recover KEY\r\n");
    PRINTF(" and display them or sent to AES engine. Follow menu\r\n");
    PRINTF("***************************************************\r\n\n");

#if PRINTFRRBLOCK
    PRINTF("Flash Type*****************************************\r\n");
    PrintMem((uint8_t*)&flashInstance, sizeof(flashInstance), 16);
    PRINTF("***************************************************\r\n\n");
#endif

    menu = mainmenu;

    while (1)
    {
      PRINTF("\n\r***************** PUF state **********************\n\r");
      PufAllowPrint(PUF);
      PufStatPrint(PUF);
      PRINTF("\n\r**************************************************\n\r");
      MenuPrint(menu);

      SCANF("%d", &selection);  // scan for number choodes by user
      selection--;  // menu starts from 0 .. so sub 1
      MenuFindFnc(menulist, sizeof(menulist)/sizeof(menulist[0]), menu, &actualfnc); // find table of functions associated to menu

      if((selection <  MenuLength(menu)) && (selection >= 0))
      {
        if(actualfnc[selection] != NULL)
          (actualfnc[selection])();
      }
      else
      {
        PRINTF("\n\rNumber %d is bad input\r\n",selection+1);
      }
    }
}


/***************************  functions  **************************************/
/************************ MAIN menu functions **************************************/

void MainEnrollPuf(void)
{
	EnrolPuf();
	menu = mainmenu;
};


void MainStartPuf(void)
{
	StartPuf();
	menu = mainmenu;
};

void MainMiscPuf(void)
{
	menu = miscmenu;
}

void MainGenerateKey(void)
{
	menu = setkeymenu;
}

void MainGetKey(void)
{
	GetKey();
	menu = mainmenu;
}

void MainAES(void)
{
    TestAesEcb();
    menu = mainmenu;
}

void MainSensorPufDemo(void)
{
    PRINTF("\n*** Starting Sensor PUF Demonstration ***\n");
    RunSensorPufDemo();
    menu = mainmenu;
}

void MainBack(void)
{
  PRINTF("\n\n\rGoing back \r\n\n");
}

/************************** MAIN menu end ********************************/
/************************** MISC menu func *******************************/
/*************************************************************************/
void MiscInitPuf(void)
{
  InitPuf();
  menu = miscmenu;
}

void MiscStopPuf(void)
{
  StopPuf();
  menu = miscmenu;
}

void MiscZeroize(void)
{
  Zeroize();
  menu = miscmenu;
}

void MiscBlockEnroll(void)
{
  BlockEnroll();
  menu = miscmenu;
}

void MiscBlockSetKey(void)
{
  BlockSetKey();
  menu = miscmenu;
}

void MiscBack(void)
{
   menu = mainmenu;
}

void KeyBack(void)
{
  menu = mainmenu;
}


/***********************************************************************/
/***********************************************************************/
uint32_t KeyCodeCheck(uint8_t * keycode, uint32_t * keytype, uint32_t * keyidx, uint32_t * keysize)
{
    *keytype = keycode[0];
    *keyidx = keycode[1];
    *keysize = keycode[3] == 0 ? 512 : 8 * keycode[3] ;
    if (keycode[0] >= 2)
      return 1;
    
    if (keycode[1] >= 16)
      return 2;

    if (keycode[3] >= 64)
      return 3;
  
    return 0;
}

void PufAllowPrint(PUF_Type *base)
{
  PRINTF("Allowed operations: Enroll  Start   SetKey  GetKey ");
  PRINTF("\r\n                    %s %s %s %s ",\
    (base->ALLOW & PUF_ALLOW_ALLOWENROLL_MASK) ? "  yes  " : "  no   ",\
    (base->ALLOW & PUF_ALLOW_ALLOWSTART_MASK)  ? "  yes  " : "  no   ",\
    (base->ALLOW & PUF_ALLOW_ALLOWSETKEY_MASK) ? "  yes  " : "  no   ",\
    (base->ALLOW & PUF_ALLOW_ALLOWGETKEY_MASK) ? "  yes  " : "  no   ");
}

/********************************************************************/
void PufStatPrint(PUF_Type *base)
{
  PRINTF("\r\n\r\nPUF Status:         Busy   Success  Error ");
  PRINTF("\r\n                    %s %s %s",\
         (base->STAT & PUF_STAT_BUSY_MASK) ?    "  yes  " : "  no   ",\
         (base->STAT & PUF_STAT_SUCCESS_MASK) ? "  yes  " : "  no   ",\
         (base->STAT & PUF_STAT_ERROR_MASK) ?   "  yes  " : "  no   ");
         
}

/********************************************************************/
void MenuPrint(char ** menu)
{
  uint32_t i = 0;
  PRINTF("\n\r");
  do
  {
      PRINTF("%d. %s\r\n",i+1,(char const *)menu[i]);
  }while(NULL == strstr((char const *)menu[i++], "Back"));
}

/********************************************************************/
uint32_t MenuLength(char ** menu)
{
  uint32_t i = 0;
  while(NULL == strstr((char const *)menu[i], "Back"))
      i++;
  
  return ++i;
}

/*************************************/
void MenuFindFnc(sMenu * menuList, uint32_t menuListLen, char ** actualmenu, void (*** fnc)(void))
{
  uint32_t i;
  for(i=0; i < menuListLen; i++)
  {
      if(menuList[i].menutext == actualmenu)
      {
          *fnc = menuList[i].menufnc;
          return; 
      }
  }
  fnc = NULL;
  return;
}

/********************************************************************/
void PrintMem(uint8_t * adr, uint32_t len, uint32_t seg)
{
        uint32_t i = 0;
        while (i < len)
        {
        
          if ((i % seg) == 0)  
            PRINTF("\n\r%4d: ",i);  
          
          PRINTF("%2x ", adr[i++]);
        }
        putchar(0xD); putchar(0xA);
}

/********************************************************************/
uint32_t IsEmptyMem(uint8_t * adr, uint32_t len)
{
      uint32_t i = 0;
      while (i < len)
      {
          if(adr[i] != 0)
            return 1;
          i++;
      }
      return 0;
}

/********************************************************************/
 void PrintKeyCode(uint8_t * kc, uint32_t size, uint32_t segmentation)
 {
    PRINTF("Key code:\r\n");
    PRINTF("        ---------- key type\r\n");
    PRINTF("       |   --------- key index\r\n");
    PRINTF("       |  |      -------- key size\r\n");
    PRINTF("       |  |     |\r\n");
    PrintMem(kc, size, segmentation);
 }

 /**********************************************************************/
 /************************ MISC FUNCTIONS ******************************/
 /**********************************************************************/


 /************************ StoreKeyCode *********************************/
 void StoreKeyCode(keyCode_t keycode, uint32_t size)
 {
     uint32_t keystore;
     PRINTF("\r\nStore key code to \r\n1. RAM keycode0\r\n2. RAM keycode1\r\n3. \
FLASH keycode0\r\n4. FLASH keycode1\r\n");
     while(1)
     {
       SCANF("%d", &keystore);
       keystore--;
       if(keystore < NUM_OF_KEYS)
       {
			memcpy(pufData.keyCode[keystore], keycode, size);
			return;
       }
       else if(keystore < NUM_OF_KEYS * 2)
       {
			Flash_StoreKC(keycode, keystore-NUM_OF_KEYS, size);
			return;
       }
       else
           PRINTF("\r\nInput value %d is bad\r\n", ++keystore);
       }
 }

 /************************ LoadKeyCode *********************************/
 void LoadKeyCode(uint8_t ** keycode)
 {
     uint32_t keystore, keyIdx;
     status_t status;
     while(1)
     {
       PRINTF("\n\rReconstruct key from keycode\r\n1. RAM keycode0\r\n2. \
RAM keycode1\n\r3. FLASH keycode0\r\n4. FLASH keycode1\r\n5. CMPA key store\r\n");
       SCANF("%d", &keystore);
       keystore--;
       if(keystore < NUM_OF_KEYS)
       {
         *keycode = &pufData.keyCode[keystore][0];

         return;
       }
       else if(keystore < NUM_OF_KEYS * 2)
       {
         Flash_ReadKC(keycode, keystore-NUM_OF_KEYS);
         return;
       }
       else if(keystore == NUM_OF_KEYS * 2)
       {

           PRINTF("\r\nEnter key code to be loaded");
           PRINTF("\r\n1. Secure Boot Key Code\r\n2. User KEK\r\n3. \
Unique Device Secret Key Code\r\n4. Princ0 Key Code");
           PRINTF("\r\n5. Princ1 Key Code\r\n6. Princ2 Key Code\r\n");

           while(1)
           {
             SCANF("%d", &keyIdx);
             keyIdx--;
             if(keyIdx < 6)
               break;
             else
               PRINTF("\r\nBad number, enter again");
           }

           status = FFR_KeystoreGetKC(&flashInstance, pufCmpaData.keyCode[0], (ffr_key_type_t)keyIdx);
           if(status == kStatus_Success)
           {
             PRINTF("\r\nGet Key Code successful\r\n");
           }
           else
           {
             PRINTF("\r\nGet Key Code failed\r\n");
           }
           *keycode = pufCmpaData.keyCode[0];

           return;
       }
       else
       {
         PRINTF("\r\nBad number entered\r\n");
         continue;
       }
     }
   }


uint32_t Flash_StoreKC(uint8_t *kcBuf, uint32_t keyIdx, uint32_t size)
{

	uint32_t failedAddress, failedData;
	sPufData pufFlMirror __attribute__ ((aligned (256)));
	uint32_t status;

	memset( (uint8_t*)&pufFlMirror, 0, sizeof(pufFlMirror) );
	memcpy((uint8_t *)&pufFlMirror, (uint8_t *)FLASHSTORE_BASEADR, sizeof(pufFlMirror));

	status = FLASH_Erase(&flashInstance, FLASHSTORE_BASEADR, FLASHSTORE_LEN, kFLASH_ApiEraseKey);
	//verify_status(status);

	status = FLASH_VerifyErase(&flashInstance, FLASHSTORE_BASEADR, FLASHSTORE_LEN);
	//verify_status(status);

	memcpy((uint8_t *)&pufFlMirror.keyCode[keyIdx][0], kcBuf, size);

	status =  FLASH_Program(&flashInstance, FLASHSTORE_BASEADR, (uint8_t*)&pufFlMirror, FLASHSTORE_LEN);
	//verify_status(status);

	status = FLASH_VerifyProgram(&flashInstance, FLASHSTORE_BASEADR, FLASHSTORE_LEN, (uint8_t *)&pufFlMirror,
					 &failedAddress, &failedData);
	//verify_status(status);

	return 0;

}

void Flash_ReadKC(uint8_t **kcBuf, uint32_t keyIdx)
{
	*kcBuf = &(((sPufData*)FLASHSTORE_BASEADR)->keyCode[keyIdx][0]);
}



uint32_t Flash_StoreAC(uint8_t *acBuf)
{
	uint32_t failedAddress, failedData;
	sPufData pufFlMirror __attribute__ ((aligned (256)));
	uint32_t status;

	memset( (uint8_t*)&pufFlMirror, 0, sizeof(pufFlMirror) );
	memcpy((uint8_t *)&pufFlMirror, (uint8_t *)FLASHSTORE_BASEADR, sizeof(pufFlMirror));

	status = FLASH_Erase(&flashInstance, FLASHSTORE_BASEADR, FLASHSTORE_LEN, kFLASH_ApiEraseKey);
	//verify_status(status);

	status = FLASH_VerifyErase(&flashInstance, FLASHSTORE_BASEADR, FLASHSTORE_LEN);
	//verify_status(status);

	memcpy((uint8_t *)&pufFlMirror.activationCode, acBuf, sizeof(pufFlMirror.activationCode));

	status =  FLASH_Program(&flashInstance, FLASHSTORE_BASEADR, (uint8_t*)&pufFlMirror, FLASHSTORE_LEN);
	//verify_status(status);

	status = FLASH_VerifyProgram(&flashInstance, FLASHSTORE_BASEADR, FLASHSTORE_LEN, (uint8_t *)&pufFlMirror,
					 &failedAddress, &failedData);
	//verify_status(status);

	return 0;
}

void Flash_ReadAC(uint8_t ** acBuf)
{
	*acBuf = &(((sPufData*)FLASHSTORE_BASEADR)->activationCode[0]);
}

void verify_status(status_t status)
{
    char *tipString = "Unknown status";
    switch (status)
    {
        case kStatus_Success:
            tipString = "Done.";
            break;
        case kStatus_InvalidArgument:
            tipString = "Invalid argument.";
            break;
        case kStatus_FLASH_AlignmentError:
            tipString = "Alignment Error.";
            break;
        case kStatus_FLASH_AccessError:
            tipString = "Flash Access Error.";
            break;
        case kStatus_FLASH_CommandNotSupported:
            tipString = "This API is not supported in current target.";
            break;


        default:
            break;
    }
    PRINTF("%s\r\n\r\n", tipString);
}


