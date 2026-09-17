/PROG  DEMO
/ATTR
OWNER		= MNEDITOR;
COMMENT		= "robotparse demo";
PROG_SIZE	= 1024;
CREATE		= DATE 26-09-16  TIME 10:00:00;
MODIFIED	= DATE 26-09-16  TIME 10:00:00;
FILE_NAME	= ;
VERSION		= 0;
LINE_COUNT	= 21;
MEMORY_SIZE	= 1500;
PROTECT		= READ_WRITE;
TCD:  STACK_SIZE	= 0,
      TASK_PRIORITY	= 50,
      TIME_SLICE	= 0,
      BUSY_LAMP_OFF	= 0,
      ABORT_REQUEST	= 0,
      PAUSE_REQUEST	= 0;
DEFAULT_GROUP	= 1,*,*,*,*;
CONTROL_CODE	= 00000000 00000000;
/APPL
/MN
   1:  !robotparse demo ;
   2:  UFRAME_NUM=1 ;
   3:  UTOOL_NUM=1 ;
   4:J P[1:HOME] 100% FINE    ;
   5:J P[2] 50% CNT50    ;
   6:L P[3] 500mm/sec FINE    ;
   7:L P[4] 150mm/sec CNT100    ;
   8:L P[5] 150mm/sec CNT50 ACC80    ;
   9:C P[6]    
    :  P[7] 150mm/sec CNT20    ;
  10:L P[7] 250mm/sec CNT50 Offset,PR[1]    ;
  11:  WAIT   0.50(sec) ;
  12:  FOR R[1]=1 TO 3 ;
  13:L P[8] 250mm/sec CNT30 Offset,PR[2]    ;
  14:  PR[2,2]=PR[2,2]+75 ;
  15:  ENDFOR ;
  16:A P[9] 200mm/sec CNT100    ;
  17:A P[10] 200mm/sec CNT100    ;
  18:A P[11] 200mm/sec FINE    ;
  19:J P[1:HOME] 100% FINE    ;
/POS
P[1:"HOME"]{
   GP1:
	UF : 1, UT : 1,	
	J1=     0.000 deg,	J2=   -90.000 deg,	J3=    90.000 deg,
	J4=     0.000 deg,	J5=    90.000 deg,	J6=     0.000 deg
};
P[2]{
   GP1:
	UF : 1, UT : 1,		CONFIG : 'N U T, 0, 0, 0',
	X =   150.000  mm,	Y =  -200.000  mm,	Z =   450.000  mm,
	W =   180.000 deg,	P =     0.000 deg,	R =     0.000 deg
};
P[3]{
   GP1:
	UF : 1, UT : 1,		CONFIG : 'N U T, 0, 0, 0',
	X =   150.000  mm,	Y =  -200.000  mm,	Z =   250.000  mm,
	W =   180.000 deg,	P =     0.000 deg,	R =     0.000 deg
};
P[4]{
   GP1:
	UF : 1, UT : 1,		CONFIG : 'N U T, 0, 0, 0',
	X =   300.000  mm,	Y =  -200.000  mm,	Z =   250.000  mm,
	W =   180.000 deg,	P =     0.000 deg,	R =     0.000 deg
};
P[5]{
   GP1:
	UF : 1, UT : 1,		CONFIG : 'N U T, 0, 0, 0',
	X =   300.000  mm,	Y =     0.000  mm,	Z =   250.000  mm,
	W =   180.000 deg,	P =     0.000 deg,	R =     0.000 deg
};
P[6]{
   GP1:
	UF : 1, UT : 1,		CONFIG : 'N U T, 0, 0, 0',
	X =   225.000  mm,	Y =    75.000  mm,	Z =   250.000  mm,
	W =   180.000 deg,	P =     0.000 deg,	R =     0.000 deg
};
P[7]{
   GP1:
	UF : 1, UT : 1,		CONFIG : 'N U T, 0, 0, 0',
	X =   150.000  mm,	Y =     0.000  mm,	Z =   250.000  mm,
	W =   180.000 deg,	P =     0.000 deg,	R =     0.000 deg
};
P[8]{
   GP1:
	UF : 1, UT : 1,		CONFIG : 'N U T, 0, 0, 0',
	X =   200.000  mm,	Y =     0.000  mm,	Z =   300.000  mm,
	W =   180.000 deg,	P =     0.000 deg,	R =     0.000 deg
};
P[9]{
   GP1:
	UF : 1, UT : 1,		CONFIG : 'N U T, 0, 0, 0',
	X =   150.000  mm,	Y =   125.000  mm,	Z =   300.000  mm,
	W =   180.000 deg,	P =     0.000 deg,	R =     0.000 deg
};
P[10]{
   GP1:
	UF : 1, UT : 1,		CONFIG : 'N U T, 0, 0, 0',
	X =   200.000  mm,	Y =   175.000  mm,	Z =   300.000  mm,
	W =   180.000 deg,	P =     0.000 deg,	R =     0.000 deg
};
P[11]{
   GP1:
	UF : 1, UT : 1,		CONFIG : 'N U T, 0, 0, 0',
	X =   250.000  mm,	Y =   125.000  mm,	Z =   300.000  mm,
	W =   180.000 deg,	P =     0.000 deg,	R =     0.000 deg
};
/END
