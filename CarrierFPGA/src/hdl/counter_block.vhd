--------------------------------------------------------------------------------
--  File:       counter_block.vhd
--  Desc:       Position compare output pulse generator
--
--------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.top_defines.all;

entity counter_block is
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
    out_o               : out std_logic_vector(31 downto 0);
    carry_o             : out std_logic
);
end counter_block;

architecture rtl of counter_block is

signal ENABLE_VAL       : std_logic_vector(31 downto 0);
signal TRIG_VAL         : std_logic_vector(31 downto 0);
signal DIR              : std_logic_vector(31 downto 0);
signal START            : std_logic_vector(31 downto 0);
signal START_WSTB       : std_logic;
signal STEP             : std_logic_vector(31 downto 0);

signal enable           : std_logic;
signal trig             : std_logic;

begin

--
-- Control System Interface
--
counter_ctrl : entity work.counter_ctrl
port map (
    clk_i               => clk_i,
    reset_i             => reset_i,
    sysbus_i            => sysbus_i,
    posbus_i            => (others => (others => '0')),
    trig_o              => trig,
    enable_o            => enable,

    mem_cs_i            => mem_cs_i,
    mem_wstb_i          => mem_wstb_i,
    mem_addr_i          => mem_addr_i,
    mem_dat_i           => mem_dat_i,

    DIR                 => DIR,
    DIR_WSTB            => open,
    START               => START,
    START_WSTB          => START_WSTB,
    STEP                => STEP,
    STEP_WSTB           => open
);

-- LUT Block Core Instantiation
counter : entity work.counter
port map (
    clk_i               => clk_i,
    reset_i             => reset_i,

    enable_i            => enable,
    trigger_i           => trig,

    DIR                 => DIR(0),
    START               => START,
    START_LOAD          => START_WSTB,
    STEP                => STEP,

    carry_o             => carry_o,
    out_o               => out_o
);

end rtl;

