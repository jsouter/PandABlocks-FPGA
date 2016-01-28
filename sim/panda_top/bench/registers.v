localparam TTL_CS                 = 0;
localparam LVDS_CS                = 1;
localparam LUT_CS                 = 2;
localparam SRGATE_CS              = 3;
localparam DIV_CS                 = 4;
localparam PULSE_CS               = 5;
localparam SEQ_CS                 = 6;
localparam INENC_CS               = 7;
localparam QDEC_CS                = 8;
localparam OUTENC_CS              = 9;
localparam POSENC                 = 10;
localparam CALC                   = 11;
localparam ADDER                  = 12;
localparam COUNTER_CS             = 13;
localparam PGEN                   = 14;
localparam PCOMP_CS               = 15;
localparam PCAP_CS                = 16;
localparam SLOW_CS                = 26;
localparam CLOCKS_CS              = 27;
localparam BITS_CS                = 28;
localparam POSITIONS_CS           = 29;
localparam DRV_CS                 = 30;
localparam REG_CS                 = 31;


localparam TTLOUT_VAL         = 0;

localparam LVDSOUT_VAL        = 0;

localparam LUT_INPA           = 0;
localparam LUT_INPB           = 1;
localparam LUT_INPC           = 2;
localparam LUT_INPD           = 3;
localparam LUT_INPE           = 4;
localparam LUT_FUNC           = 5;

localparam SRGATE_SET         = 0;
localparam SRGATE_RST         = 1;
localparam SRGATE_SET_EDGE    = 2;
localparam SRGATE_RST_EDGE    = 3;
localparam SRGATE_FORCE_SET   = 4;
localparam SRGATE_FORCE_RST   = 5;

localparam DIV_INP            = 0;
localparam DIV_RST            = 1;
localparam DIV_DIVISOR        = 2;
localparam DIV_FIRST_PULSE    = 3;
localparam DIV_COUNT          = 4;
localparam DIV_FORCE_RST      = 5;

localparam PULSE_INP              = 0;
localparam PULSE_RST              = 1;
localparam PULSE_DELAY_L          = 2;
localparam PULSE_DELAY_H          = 3;
localparam PULSE_WIDTH_L          = 4;
localparam PULSE_WIDTH_H          = 5;
localparam PULSE_FORCE_RST        = 6;
localparam PULSE_ERR_OVERFLOW     = 7;
localparam PULSE_ERR_PERIOD       = 8;
localparam PULSE_QUEUE            = 9;
localparam PULSE_MISSED_CNT       = 10;

localparam SEQ_GATE               = 0;
localparam SEQ_INPA               = 1;
localparam SEQ_INPB               = 2;
localparam SEQ_INPC               = 3;
localparam SEQ_INPD               = 4;
localparam SEQ_PRESCALE           = 5;
localparam SEQ_SOFT_GATE          = 6;
localparam SEQ_TABLE_LENGTH       = 7;
localparam SEQ_TABLE_CYCLE        = 8;
localparam SEQ_CUR_FRAME          = 9;
localparam SEQ_CUR_FCYCLE         = 10;
localparam SEQ_CUR_TCYCLE         = 11;
localparam SEQ_TABLE_STROBES      = 12;
localparam SEQ_TABLE_RST          = 13;
localparam SEQ_TABLE_DATA         = 14;

localparam INENC_PROTOCOL    = 0;
localparam INENC_CLKRATE     = 1;
localparam INENC_FRAMERATE   = 2;
localparam INENC_BITS        = 3;
localparam INENC_SETP        = 4;
localparam INENC_RST_ON_Z    = 5;

localparam OUTENC_A      = 0;
localparam OUTENC_B      = 1;
localparam OUTENC_Z      = 2;
localparam OUTENC_CONN   = 3;
localparam OUTENC_POSN   = 4;
localparam OUTENC_PROTOCOL   = 5;
localparam OUTENC_BITS       = 6;
localparam OUTENC_QPRESCALAR = 7;
localparam OUTENC_FRC_QSTATE = 8;
localparam OUTENC_QSTATE     = 9;

localparam COUNTER_ENABLE         = 0;
localparam COUNTER_TRIGGER        = 1;
localparam COUNTER_DIR            = 2;
localparam COUNTER_START          = 3;
localparam COUNTER_STEP           = 4;

localparam PCAP_ENABLE            = 0;
localparam PCAP_FRAME             = 1;
localparam PCAP_CAPTURE           = 2;
localparam PCAP_MISSED_CAPTURES   = 3;
localparam PCAP_ERR_STATUS        = 4;

localparam PCOMP_ENABLE       = 0;
localparam PCOMP_POSN         = 1;
localparam PCOMP_START        = 2;
localparam PCOMP_STEP         = 3;
localparam PCOMP_WIDTH        = 4;
localparam PCOMP_NUMBER       = 5;
localparam PCOMP_RELATIVE     = 6;
localparam PCOMP_DIR          = 7;
localparam PCOMP_FLTR_DELTAT  = 8;
localparam PCOMP_FLTR_THOLD   = 9;
localparam PCOMP_LUT_ENABLE   = 10;

localparam CLOCKS_A_PERIOD        = 0;
localparam CLOCKS_B_PERIOD        = 1;
localparam CLOCKS_C_PERIOD        = 2;
localparam CLOCKS_D_PERIOD        = 3;

localparam BITS_A_SET             = 0;
localparam BITS_B_SET             = 1;
localparam BITS_C_SET             = 2;
localparam BITS_D_SET             = 3;

localparam SLOW_INENC_CTRL        = 0;
localparam SLOW_OUTENC_CTRL       = 1;
localparam SLOW_VERSION           = 2;

localparam REG_BIT_READ_RST       = 0;
localparam REG_BIT_READ_VALUE     = 1;
localparam REG_POS_READ_RST       = 2;
localparam REG_POS_READ_VALUE     = 3;
localparam REG_POS_READ_CHANGES   = 4;
localparam REG_PCAP_START_WRITE   = 5;
localparam REG_PCAP_WRITE         = 6;
localparam REG_PCAP_FRAMING_MASK  = 7;
localparam REG_PCAP_FRAMING_ENABLE= 8;
localparam REG_PCAP_FRAMING_MODE  = 9;
localparam REG_PCAP_ARM           = 10;
localparam REG_PCAP_DISARM        = 11;

localparam DRV_PCAP_DMAADDR       = 0;
localparam DRV_PCAP_BLOCK_SIZE    = 1;
localparam DRV_PCAP_TIMEOUT       = 2;
localparam DRV_PCAP_IRQ_STATUS    = 3;
localparam DRV_PCAP_SMPL_COUNT    = 4;


// Panda Base Address
localparam BASE                 = 32'h43C0_0000;
localparam INENC_BASE           = BASE + 4096 * INENC_CS;
localparam PCOMP_BASE           = BASE + 4096 * PCOMP_CS;
localparam PCAP_BASE            = BASE + 4096 * PCAP_CS;
localparam OUTENC_BASE          = BASE + 4096 * OUTENC_CS;
localparam SLOW_BASE            = BASE + 4096 * SLOW_CS;
localparam CLOCKS_BASE          = BASE + 4096 * CLOCKS_CS;
localparam COUNTER_BASE         = BASE + 4096 * COUNTER_CS;
localparam DRV_BASE             = BASE + 4096 * DRV_CS;
localparam REG_BASE             = BASE + 4096 * REG_CS;

// Block Registers


