--------------------------------------------------------------------------------
--  PandA Motion Project - 2016
--      Diamond Light Source, Oxford, UK
--      SOLEIL Synchrotron, GIF-sur-YVETTE, France
--
--  Author      : Dr. Isa Uzun (isa.uzun@diamond.ac.uk)
--------------------------------------------------------------------------------
--
--  Description : Top-level design instantiating 4 channels of INENC block.
--
--------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library unisim;
use unisim.vcomponents.all;

library work;
use work.support.all;
use work.top_defines.all;

entity inenc_top is
port (
    -- Clock and Reset
    clk_i               : in  std_logic;
    reset_i             : in  std_logic;
    -- Memory Bus Interface
    mem_addr_i          : in  std_logic_vector(PAGE_AW-1 downto 0);
    mem_cs_i            : in  std_logic;
    mem_wstb_i          : in  std_logic;
    mem_dat_i           : in  std_logic_vector(31 downto 0);
    mem_dat_o           : out std_logic_vector(31 downto 0);
    -- Encoder I/O Pads
    A_IN                : in  std_logic_vector(ENC_NUM-1 downto 0);
    B_IN                : in  std_logic_vector(ENC_NUM-1 downto 0);
    Z_IN                : in  std_logic_vector(ENC_NUM-1 downto 0);
    CLK_OUT             : out std_logic_vector(ENC_NUM-1 downto 0);
    DATA_IN             : in  std_logic_vector(ENC_NUM-1 downto 0);
    CONN_OUT            : out std_logic_vector(ENC_NUM-1 downto 0);
    -- Block Outputs
    DCARD_MODE          : in  std32_array(ENC_NUM-1 downto 0);
    PROTOCOL            : out std3_array(ENC_NUM-1 downto 0);
    posn_o              : out std32_array(ENC_NUM-1 downto 0);
    posn_upper_o        : out std32_array(ENC_NUM-1 downto 0)
);
end inenc_top;

architecture rtl of inenc_top is

signal mem_blk_cs       : std_logic_vector(ENC_NUM-1 downto 0);

begin

-- Unused outputs.
mem_dat_o <= (others => '0');

--
-- Instantiate INENC Blocks :
--  There are ENC_NUM amount of encoders on the board
--
INENC_GEN : FOR I IN 0 TO ENC_NUM-1 GENERATE

-- Generate Block chip select signal
mem_blk_cs(I) <= '1'
    when (mem_addr_i(PAGE_AW-1 downto BLK_AW) = TO_SVECTOR(I, PAGE_AW-BLK_AW)
            and mem_cs_i = '1') else '0';

inenc_block_inst : entity work.inenc_block
port map (

    clk_i               => clk_i,
    reset_i             => reset_i,

    mem_cs_i            => mem_blk_cs(I),
    mem_wstb_i          => mem_wstb_i,
    mem_addr_i          => mem_addr_i(BLK_AW-1 downto 0),
    mem_dat_i           => mem_dat_i,

    a_i                 => A_IN(I),
    b_i                 => B_IN(I),
    z_i                 => Z_IN(I),
    mclk_o              => CLK_OUT(I),
    mdat_i              => DATA_IN(I),
    mdat_o              => open,
    conn_o              => CONN_OUT(I),

    DCARD_MODE          => DCARD_MODE(I),
    PROTOCOL            => PROTOCOL(I),
    posn_o              => posn_o(I),
    posn_upper_o        => posn_upper_o(I)
);

END GENERATE;

end rtl;

