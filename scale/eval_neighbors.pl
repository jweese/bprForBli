#!/usr/bin/perl -w
use strict;

my %scores;

while (<>) {
  chomp;
  my @fields = split /,/;
  my ($src,$score) = @fields[0,2];
  next unless defined $score and $score =~ /^\d+$/;
  my $top = $scores{$src} // 0;
  $scores{$src} = $score if $score >= $top;
}

while (my ($word, $top) = each %scores) {
  printf "%s\t%d\n", $word, $top;
}
