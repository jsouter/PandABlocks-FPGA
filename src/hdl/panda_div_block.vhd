--------------------------------------------------------------------------------
--  File:       panda_div_block.vhd
--  Desc:       Position compare output pulse generator
--
--------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.type_defines.all;
use work.addr_defines.all;
use work.top_defines.all;

entity panda_div_block is
port (
    -- Clock and Reset
    clk_i               : in  std_logic;
    reset_i             : in  std_logic;
    -- Memory Bus Interface
    mem_cs_i            : in  std_logic;
    mem_wstb_i          : in  std_logic;
    mem_addr_i          : in  std_logic_vector(BLK_AW-1 downto 0);
    mem_dat_i           : in  std_logic_vector(31 downto 0);
    mem_dat_o           : out std_logic_vector(31 downto 0);
    -- Block inputs
    sysbus_i            : in  sysbus_t;
    -- Output pulse
    outd_o              : out std_logic;
    outn_o              : out std_logic
);
end panda_div_block;

architecture rtl of panda_div_block is

signal INP_VAL      : std_logic_vector(SBUSBW-1 downto 0);
signal RST_VAL      : std_logic_vector(SBUSBW-1 downto 0);
signal FIRST_PULSE  : std_logic := '0';
signal DIVISOR      : std_logic_vector(31 downto 0);
signal COUNT        : std_logic_vector(31 downto 0);
signal FORCE_RST    : std_logic;

signal inp          : std_logic;
signal rst          : std_logic;

signal mem_addr         : natural range 0 to (2**mem_addr_i'length - 1);

begin

-- Integer conversion for address.
mem_addr <= to_integer(unsigned(mem_addr_i));

--
-- Control System Interface
--
REG_WRITE : process(clk_i)
begin
    if rising_edge(clk_i) then
        if (reset_i = '1') then
            INP_VAL <= TO_SVECTOR(0, SBUSBW);
            RST_VAL <= TO_SVECTOR(0, SBUSBW);
            FIRST_PULSE <= '0';
            DIVISOR <= (others => '0');
            FORCE_RST <= '0';
        else
            FORCE_RST <= '0';

            if (mem_cs_i = '1' and mem_wstb_i = '1') then
                -- Input Select Control Registers
                if (mem_addr = DIV_INP) then
                    INP_VAL <= mem_dat_i(SBUSBW-1 downto 0);
                end if;

                if (mem_addr = DIV_RST) then
                    RST_VAL <= mem_dat_i(SBUSBW-1 downto 0);
                end if;

                if (mem_addr = DIV_FIRST_PULSE) then
                    FIRST_PULSE <= mem_dat_i(0);
                end if;

                if (mem_addr = DIV_DIVISOR) then
                    DIVISOR <= mem_dat_i;
                end if;

                if (mem_addr = DIV_FORCE_RST) then
                    FORCE_RST <= '1';
                end if;
            end if;
        end if;
    end if;
end process;

-- There is only 1 status register to read so no need to waste
-- a case statement.
REG_READ : process(clk_i)
begin
    if rising_edge(clk_i) then
        mem_dat_o <= COUNT;
    end if;
end process;

--
-- Core Input Port Assignments
--
process(clk_i)
begin
    if rising_edge(clk_i) then
        inp <= SBIT(sysbus_i, INP_VAL);
        rst <= SBIT(sysbus_i, RST_VAL);
    end if;
end process;


-- LUT Block Core Instantiation
panda_div : entity work.panda_div
port map (
    clk_i               => clk_i,
    reset_i             => reset_i,

    inp_i               => inp,
    rst_i               => rst,
    outd_o              => outd_o,
    outn_o              => outn_o,

    FIRST_PULSE         => FIRST_PULSE,
    DIVISOR             => DIVISOR,
    FORCE_RST           => FORCE_RST,

    COUNT               => COUNT
);

end rtl;

