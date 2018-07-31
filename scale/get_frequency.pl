#!/usr/bin/perl -w
use strict;

my $corpus = shift;
open CORPUS, "<$corpus" or die "$!";

my %voc;
while (<CORPUS>) {
  chomp;
  $voc{$_}++ foreach split;
}
close CORPUS;

while (<>) {
  chomp;
  my ($source) = split /\t/;
  my $count = $voc{$source} // 0;
  printf "%s\t%d\n", $_, $count;
}
